"""Weather network regressions without a Kodi process or external HTTP service."""

from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import sys
from threading import Event, Lock, Thread
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import requests


ADDON_DIR = Path(__file__).resolve().parents[1] / 'weather.gismeteo'
LOCATION_XML = (
    b'<locations><city n="Test City" id="1" country_name="Test Country" '
    b'lat="0" lng="0" kind="C" /></locations>'
)


class StubAddon:
    id = 'weather.gismeteo'

    def __init__(self, addon_id=''):
        self.id = addon_id or self.id

    def get_setting(self, _name):
        return 0

    def initialize_gettext(self):
        return str

    def log_debug(self, _message):
        pass

    def log_error(self, _message):
        pass


class StubPlugin(StubAddon):
    def action(self, _name=None):
        return lambda func: func

    def mem_cached(self, _seconds):
        return lambda func: func


def kodi_stubs():
    xbmc = ModuleType('xbmc')
    xbmc.Monitor = Mock()
    xbmc.getInfoLabel = lambda _label: '21.0'
    xbmc.getRegion = lambda _name: ''
    xbmc.getLanguage = lambda: 'English'
    xbmc.getCondVisibility = lambda _condition: False
    xbmc.getUserAgent = lambda: 'Kodi weather test'

    xbmcgui = ModuleType('xbmcgui')
    xbmcgui.Window = Mock()

    simpleplugin = ModuleType('simpleplugin')
    simpleplugin.Addon = StubAddon
    simpleplugin.Plugin = StubPlugin
    simpleplugin.py2_encode = simpleplugin.py2_decode = lambda value: value
    simpleplugin.translate_path = str
    return {module.__name__: module for module in (xbmc, xbmcgui, simpleplugin)}


class WeatherRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        with self.server.request_lock:
            self.server.request_count += 1
            first_request = self.server.request_count == 1

        # Send no headers on the first connection. The fallback response bounds
        # the test even if a regression removes the client's read timeout.
        if first_request and self.server.first_response == 'stall' and self.server.release_first.wait(timeout=3):
            return

        content_length = len(LOCATION_XML)
        if first_request and self.server.first_response == 'truncated':
            # Keep the XML valid so only the incomplete HTTP body triggers retry.
            content_length += 8

        self.send_response(200)
        self.send_header('Content-Type', 'application/xml')
        self.send_header('Content-Length', str(content_length))
        self.end_headers()
        self.wfile.write(LOCATION_XML)

    def log_message(self, _format, *_args):
        pass


class WeatherNetworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        module_patch = patch.dict(sys.modules, kodi_stubs())
        module_patch.start()
        cls.addClassCleanup(module_patch.stop)
        path_patch = patch.object(sys, 'path', [str(ADDON_DIR), *sys.path])
        path_patch.start()
        cls.addClassCleanup(path_patch.stop)

        spec = importlib.util.spec_from_file_location('weather_network_addon', ADDON_DIR / 'default.py')
        cls.addon = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.addon)

    def setUp(self):
        self.monitor = Mock()
        self.monitor.abortRequested.return_value = False
        self.monitor.waitForAbort.return_value = False
        monitor_patch = patch.object(self.addon.xbmc, 'Monitor', return_value=self.monitor)
        monitor_patch.start()
        self.addCleanup(monitor_patch.stop)

    @contextmanager
    def loopback_client(self, first_response):
        with ThreadingHTTPServer(('127.0.0.1', 0), WeatherRequestHandler) as server:
            server.daemon_threads = False
            server.request_lock = Lock()
            server.request_count = 0
            server.first_response = first_response
            server.release_first = Event()
            thread = Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
            thread.start()
            try:
                client = self.addon.Gismeteo('en')
                self.addCleanup(client._client.close)
                client._client.trust_env = False
                host, port = server.server_address
                client._base_url = f'http://{host}:{port}'

                with patch.object(type(client), 'REQUEST_TIMEOUT', (1, 0.5)):
                    yield client, server
            finally:
                server.release_first.set()
                server.shutdown()
                thread.join(timeout=1)

    def test_stalled_http_response_times_out_and_retries(self):
        self.assert_retries_after_incomplete_response('stall')

    def test_truncated_http_body_retries(self):
        self.assert_retries_after_incomplete_response('truncated')

    def assert_retries_after_incomplete_response(self, first_response):
        with self.loopback_client(first_response) as (client, server):
            result = self.addon._call_method(client.cities_ip)

            self.assertEqual([(item['id'], item['name']) for item in result], [('1', 'Test City')])
            with server.request_lock:
                self.assertEqual(server.request_count, 2)
            self.monitor.waitForAbort.assert_called_once_with(1)

    def test_persistent_failure_attempts_only_twice(self):
        failure = self.addon.WebClientError(requests.ReadTimeout('test timeout'))
        request = Mock(side_effect=failure)

        with self.assertRaises(self.addon.WebClientError) as caught:
            self.addon._call_method(request)

        self.assertIs(caught.exception, failure)
        self.assertEqual(request.call_count, 2)
        self.monitor.waitForAbort.assert_called_once_with(1)

    def test_success_does_not_wait(self):
        request = Mock(return_value=['forecast'])

        result = self.addon._call_method(request, {'city_id': '1'})

        self.assertEqual(result, ['forecast'])
        request.assert_called_once_with(city_id='1')
        self.monitor.waitForAbort.assert_not_called()

    def test_abort_during_retry_delay_stops_before_second_request(self):
        request = Mock(side_effect=self.addon.GismeteoError('first request failed'))

        def abort_during_wait(_seconds):
            self.monitor.abortRequested.return_value = True
            return True

        self.monitor.waitForAbort.side_effect = abort_during_wait

        with self.assertRaises(self.addon._WeatherAborted):
            self.addon._call_method(request)

        request.assert_called_once_with()
        self.monitor.waitForAbort.assert_called_once_with(1)

    def test_abort_during_successful_request_discards_result(self):
        def finish_request():
            self.monitor.abortRequested.return_value = True
            return ['forecast']

        request = Mock(side_effect=finish_request)

        with self.assertRaises(self.addon._WeatherAborted):
            self.addon._call_method(request)

        request.assert_called_once_with()
        self.monitor.waitForAbort.assert_not_called()

    def test_abort_during_final_request_suppresses_error(self):
        failure = self.addon.WebClientError(requests.ReadTimeout('test timeout'))

        def finish_request():
            if request.call_count == 2:
                self.monitor.abortRequested.return_value = True
            raise failure

        request = Mock(side_effect=finish_request)

        with self.assertRaises(self.addon._WeatherAborted):
            self.addon._call_method(request)

        self.assertEqual(request.call_count, 2)
        self.monitor.waitForAbort.assert_called_once_with(1)

    def test_empty_auto_location_clears_displayed_name(self):
        weather = Mock()
        settings = {'CurrentLocation': True, 'Location1': 'Saved City', 'Location1ID': '1'}
        weather.get_setting.side_effect = lambda key: settings.get(key, '')
        weather.gismeteo_lang.return_value = 'en'

        with patch.object(self.addon, 'weather', weather), patch.object(
            self.addon, '_ip_locations', return_value=[]
        ) as lookup, patch.object(self.addon, '_location_forecast') as forecast_lookup:
            self.addon.forecast(SimpleNamespace(id='1'))

        lookup.assert_called_once_with('en')
        forecast_lookup.assert_not_called()
        weather.set_property.assert_any_call('Location1', '')
        weather.set_property.assert_any_call('Location2', 'Saved City')
        weather.set_property.assert_any_call('Locations', 2)
        weather.notify_error.assert_not_called()

    def test_forecast_action_treats_abort_as_normal_shutdown(self):
        weather = Mock()
        weather.get_setting.return_value = True

        with patch.object(self.addon, 'weather', weather), patch.object(
            self.addon, '_ip_locations', side_effect=self.addon._WeatherAborted
        ) as lookup:
            self.addon.forecast(SimpleNamespace(id='1'))

        lookup.assert_called_once()
        weather.notify_error.assert_not_called()
        weather.set_property.assert_not_called()


if __name__ == '__main__':
    unittest.main()
