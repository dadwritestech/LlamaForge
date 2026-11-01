import conftest_paths  # noqa: F401
import os
import tempfile
import unittest
from unittest import mock

import router_ctl
import routes
import slots
from routes import Req


def spawn(pool=None, slot_save=False, makedirs=None, keep=True):
    with mock.patch.object(router_ctl.os.path, "exists", return_value=True), \
         mock.patch.object(router_ctl, "is_running", return_value=False), \
         mock.patch.object(router_ctl, "supports_cors_origins", return_value=True), \
         mock.patch.object(router_ctl, "supports_slot_save", return_value=slot_save), \
         mock.patch.object(router_ctl.os, "makedirs", side_effect=makedirs), \
         mock.patch("builtins.open", mock.mock_open()), \
         mock.patch.object(router_ctl.procs, "write_pid"), \
         mock.patch.object(router_ctl.subprocess, "Popen") as popen:
        ok, err = router_ctl.start("server", "models.ini", 8080, "127.0.0.1", "", "logs",
                                   "L" * 43, pool, slot_save=keep)
    assert ok, err
    return popen.call_args


class Start(unittest.TestCase):
    def test_slot_save_path_when_the_build_has_it(self):
        argv = spawn(slot_save=True).args[0]
        path = argv[argv.index("--slot-save-path") + 1]
        self.assertEqual(os.path.basename(path), "slot-saves")

    def test_unusable_slot_save_dir_never_blocks_the_router(self):
        def md(path, *a, **k):
            if os.path.basename(path) == "slot-saves":
                raise OSError("a file is in the way")
        argv = spawn(slot_save=True, makedirs=md).args[0]
        self.assertNotIn("--slot-save-path", argv)

    def test_no_slot_save_path_on_builds_without_it(self):
        self.assertNotIn("--slot-save-path", spawn().args[0])

    def test_no_slot_save_path_unless_asked(self):
        # Elastic VRAM off: API-key holders get no way to write slot dumps to disk
        argv = spawn(slot_save=True, keep=False).args[0]
        self.assertNotIn("--slot-save-path", argv)
        with mock.patch.object(router_ctl.os.path, "exists", return_value=True), \
             mock.patch.object(router_ctl, "is_running", return_value=False), \
             mock.patch.object(router_ctl, "supports_slot_save", return_value=True), \
             mock.patch.object(router_ctl, "supports_cors_origins", return_value=True), \
             mock.patch("builtins.open", mock.mock_open()), \
             mock.patch.object(router_ctl.procs, "write_pid"), \
             mock.patch.object(router_ctl.subprocess, "Popen") as popen:
            router_ctl.start("server", "models.ini", 8080, "127.0.0.1", "", "logs", "L" * 43)
        self.assertNotIn("--slot-save-path", popen.call_args.args[0])   # off by default

    def test_restart_passes_slot_save_on(self):
        seen = []
        with mock.patch.object(router_ctl.network_policy, "start_error", return_value=""), \
             mock.patch.object(router_ctl, "stop", return_value=True), \
             mock.patch.object(router_ctl, "start",
                               side_effect=lambda *a, **k: seen.append(k) or (True, "")):
            router_ctl.restart("server", "models.ini", 8080, "127.0.0.1", "", "logs", "",
                               None, slot_save=True)
        self.assertEqual(seen, [{"slot_save": True}])

    def test_single_mode_is_unchanged(self):
        argv = spawn().args[0]
        self.assertEqual(argv[argv.index("--models-max") + 1], "1")
        self.assertNotIn("--no-models-autoload", argv)

    def test_pool_raises_the_cap_and_stops_autoload(self):
        argv = spawn({"models_max": 3, "autoload": False}).args[0]
        self.assertEqual(argv[argv.index("--models-max") + 1], "3")
        self.assertIn("--no-models-autoload", argv)

    def test_pool_may_keep_autoload(self):
        argv = spawn({"models_max": 2, "autoload": True}).args[0]
        self.assertNotIn("--no-models-autoload", argv)

    def test_pool_margin_becomes_the_routers_memory_margin(self):
        argv = spawn({"models_max": 3, "autoload": True, "memory_margin_mib": 1536}).args[0]
        self.assertEqual(argv[argv.index("--models-memory-margin") + 1], "1536")

    def test_no_memory_margin_without_one_in_the_pool(self):
        self.assertNotIn("--models-memory-margin", spawn({"models_max": 3, "autoload": True}).args[0])
        self.assertNotIn("--models-memory-margin", spawn().args[0])

    def test_device_order_is_left_alone(self):
        # forcing CUDA_DEVICE_ORDER would re-aim every device / main-gpu pin
        call = spawn({"models_max": 3, "autoload": False})
        env = call.kwargs.get("env")
        self.assertTrue(env is None or "CUDA_DEVICE_ORDER" not in env
                        or env["CUDA_DEVICE_ORDER"] == router_ctl.os.environ.get("CUDA_DEVICE_ORDER"))

    def test_restart_passes_the_pool_on(self):
        seen = []
        with mock.patch.object(router_ctl.network_policy, "start_error", return_value=""), \
             mock.patch.object(router_ctl, "stop", return_value=True), \
             mock.patch.object(router_ctl, "start",
                               side_effect=lambda *a, **k: seen.append(a) or (True, "")):
            router_ctl.restart("server", "models.ini", 8080, "127.0.0.1", "", "logs", "",
                               {"models_max": 3, "autoload": False})
        self.assertEqual(seen[0][-1], {"models_max": 3, "autoload": False})


class Sidecar(unittest.TestCase):
    """run.ps1 / run.sh start the router single; only a router LlamaForge
    started itself, still under the recorded PID, is known to run a pool."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_start_records_its_pool(self):
        with mock.patch.object(router_ctl, "record_pool") as rec:
            call = spawn({"models_max": 3, "autoload": False})
        self.assertEqual(rec.call_args.args[2], {"models_max": 3, "autoload": False})
        self.assertIsNotNone(call)

    def test_pool_of_the_recorded_router(self):
        router_ctl.procs.write_pid(self.dir, "router", 4242)
        router_ctl.record_pool(self.dir, 4242, {"models_max": 3, "autoload": False})
        self.assertEqual(router_ctl.running_pool(self.dir), {"models_max": 3, "autoload": False})

    def test_a_router_the_runner_started_is_single(self):
        router_ctl.record_pool(self.dir, 4242, {"models_max": 3, "autoload": False})
        router_ctl.procs.write_pid(self.dir, "router", 5151)     # run.ps1 started a new one
        self.assertIsNone(router_ctl.running_pool(self.dir))

    def test_nothing_recorded_is_single(self):
        self.assertIsNone(router_ctl.running_pool(self.dir))
        with open(os.path.join(self.dir, "router.pool.json"), "w") as f:
            f.write("{nope")
        router_ctl.procs.write_pid(self.dir, "router", 4242)
        self.assertIsNone(router_ctl.running_pool(self.dir))


POOL = {"models_max": 3, "autoload": False}


class RoutesBase(unittest.TestCase):
    def setUp(self):
        self.cfg = {"router_port": 8080, "router_host": "127.0.0.1", "router_api_key": "",
                    "router_local_key": "L" * 43, "server_bin": "/bin/llama-server",
                    "active_engine": "llamacpp", "multi_model": True, "slot_cap": 3}
        self.saved = {}

        def update(ch):
            self.cfg.update(ch)
            self.saved.update(ch)
            return dict(self.cfg)

        mock.patch.object(routes, "cfg", side_effect=lambda: dict(self.cfg)).start()
        mock.patch.object(routes.config, "update", side_effect=update).start()
        mock.patch.object(routes.config, "ini_path", return_value="/tmp/models.ini").start()
        mock.patch.object(routes.os.path, "exists", return_value=True).start()
        mock.patch.object(routes.router_ctl, "supports_no_autoload", return_value=True).start()
        mock.patch.object(routes.router_ctl, "supports_router_mode", return_value=True).start()
        self.margin = mock.patch.object(routes.router_ctl, "supports_models_memory_margin",
                                        return_value=False).start()
        self.running = mock.patch.object(routes.router_ctl, "is_running",
                                         return_value=True).start()
        self.pool_now = mock.patch.object(routes.router_ctl, "running_pool",
                                          return_value=None).start()
        self.restart = mock.patch.object(routes.router_ctl, "restart",
                                         return_value=(True, "")).start()
        self.slots = mock.patch.object(routes, "SLOTS").start()   # never the real models.ini
        self.addCleanup(mock.patch.stopall)


class ReconcilePool(RoutesBase):
    def test_a_single_router_restarts_into_the_pool(self):
        self.assertTrue(routes.reconcile_router_pool())
        self.assertEqual(self.restart.call_args.args[-1], POOL)
        self.assertFalse(self.restart.call_args.kwargs["slot_save"])   # Elastic is off

    def test_the_slot_save_folder_only_while_elastic_keeps_conversations(self):
        for on, keep, want in ((True, True, True), (True, False, False), (False, True, False)):
            self.cfg.update(elastic_vram=on, elastic_keep_conversation=keep)
            self.assertIs(routes._el_slot_save(dict(self.cfg)), want)
        self.cfg["elastic_vram"] = True
        self.cfg.pop("elastic_keep_conversation")
        self.assertTrue(routes._el_slot_save(dict(self.cfg)))   # keep defaults to on
        self.assertTrue(routes.reconcile_router_pool())
        self.assertTrue(self.restart.call_args.kwargs["slot_save"])

    def test_a_matching_router_is_left_alone(self):
        self.pool_now.return_value = dict(POOL)
        self.assertFalse(routes.reconcile_router_pool())
        self.restart.assert_not_called()

    def test_turning_it_off_goes_back_to_single(self):
        self.cfg["multi_model"] = False
        self.pool_now.return_value = dict(POOL)
        self.assertTrue(routes.reconcile_router_pool())
        self.assertIsNone(self.restart.call_args.args[-1])

    def test_default_users_see_no_restart(self):
        self.cfg["multi_model"] = False
        self.assertFalse(routes.reconcile_router_pool())
        self.restart.assert_not_called()
        self.running.assert_not_called()          # and no wait for the port

    def test_waits_for_the_runners_router_to_bind(self):
        self.running.side_effect = [False, False, True, True]
        with mock.patch.object(routes.time, "sleep"):
            self.assertTrue(routes.reconcile_router_pool())
        self.assertEqual(self.restart.call_args.args[-1], POOL)

    def test_a_router_that_is_down_is_the_runners_job(self):
        self.running.return_value = False
        self.assertFalse(routes.reconcile_router_pool(wait_s=0))
        self.restart.assert_not_called()


class RestartsKeepThePool(RoutesBase):
    def test_engine_switch(self):
        routes.post_engine_switch(Req(body={"engine": "llamacpp"}))
        self.assertEqual(self.restart.call_args.args[-1], POOL)

    def test_auth_reconcile(self):
        with mock.patch.object(routes.router_ctl, "auth_state", return_value="open"):
            routes.reconcile_router_auth()
        self.assertEqual(self.restart.call_args.args[-1], POOL)

    def test_single_mode_passes_none(self):
        self.cfg["multi_model"] = False
        routes.post_engine_switch(Req(body={"engine": "llamacpp"}))
        self.assertIsNone(self.restart.call_args.args[-1])


class SlotSettings(RoutesBase):
    def test_writable_and_checked(self):
        status, out = routes.post_config(Req(body={
            "multi_model": True, "slot_cap": 4, "slot_headroom_mib": 2048,
            "slot_autoload": False}))
        self.assertEqual(status, 200)
        self.assertEqual(self.saved["slot_cap"], 4)
        for body in ({"slot_cap": 5}, {"slot_cap": 1}, {"slot_cap": "3"}, {"slot_cap": True},
                     {"slot_headroom_mib": -1}, {"slot_headroom_mib": 1 << 20},
                     {"multi_model": "yes"}):
            with self.assertRaises(routes.ApiError, msg=str(body)):
                routes.post_config(Req(body=body))

    def test_a_pool_change_restarts_the_router(self):
        self.cfg["multi_model"] = False
        status, out = routes.post_config(Req(body={"multi_model": True}))
        self.assertEqual(self.restart.call_args.args[-1], POOL)
        self.assertEqual(out["router"], {"restarted": True, "error": ""})

    def test_headroom_alone_does_not(self):
        # the planner reads headroom live; the router only cares in autoload mode
        self.pool_now.return_value = POOL
        routes.post_config(Req(body={"slot_headroom_mib": 2048}))
        self.restart.assert_not_called()

    def test_headroom_is_the_routers_margin_in_autoload_mode(self):
        self.cfg.update(slot_autoload=True, slot_headroom_mib=1536)
        self.margin.return_value = True
        self.pool_now.return_value = {"models_max": 3, "autoload": True, "memory_margin_mib": 1536}
        routes.post_config(Req(body={"slot_headroom_mib": 2048}))
        self.assertEqual(self.restart.call_args.args[-1],
                         {"models_max": 3, "autoload": True, "memory_margin_mib": 2048})

    def test_unrelated_keys_never_restart(self):
        # the router may be single (the runner's) with multi_model on; a theme
        # change must not be what unloads the user's model
        routes.post_config(Req(body={"theme": "dark"}))
        self.restart.assert_not_called()


class RouterPool(unittest.TestCase):
    def test_off_is_none(self):
        self.assertIsNone(slots.router_pool({"multi_model": False}, True))
        self.assertIsNone(slots.router_pool({}, True))

    def test_on(self):
        self.assertEqual(slots.router_pool({"multi_model": True, "slot_cap": 3}, True),
                         {"models_max": 3, "autoload": False})

    def test_cap_is_clamped(self):
        self.assertEqual(slots.router_pool({"multi_model": True, "slot_cap": 9}, True)["models_max"], 4)
        self.assertEqual(slots.router_pool({"multi_model": True, "slot_cap": 1}, True)["models_max"], 2)
        self.assertEqual(slots.router_pool({"multi_model": True, "slot_cap": "x"}, True)["models_max"], 3)

    def test_autoload_opt_in(self):
        self.assertTrue(slots.router_pool({"multi_model": True, "slot_autoload": True},
                                          True)["autoload"])

    def test_a_build_that_cannot_stop_autoload_stays_single(self):
        # without --no-models-autoload any client request could load a model
        # behind the planner's back and the router would evict by LRU
        self.assertIsNone(slots.router_pool({"multi_model": True}, False))
        self.assertEqual(slots.router_pool({"multi_model": True, "slot_autoload": True}, False),
                         {"models_max": 3, "autoload": True})

    def test_autoload_gets_a_memory_margin_on_builds_that_have_it(self):
        # client requests load models behind the planner's back; the router's own
        # --models-memory-margin unloads by free VRAM instead of running out
        c = {"multi_model": True, "slot_autoload": True, "slot_headroom_mib": 2048}
        self.assertEqual(slots.router_pool(c, True, margin_supported=True),
                         {"models_max": 3, "autoload": True, "memory_margin_mib": 2048})
        self.assertNotIn("memory_margin_mib", slots.router_pool(c, True))

    def test_planner_mode_needs_no_router_margin(self):
        # autoload off: the planner already keeps slot_headroom_mib free
        c = {"multi_model": True, "slot_headroom_mib": 2048}
        self.assertNotIn("memory_margin_mib", slots.router_pool(c, True, margin_supported=True))

    def test_margin_defaults_and_zero_disables(self):
        c = {"multi_model": True, "slot_autoload": True}
        self.assertEqual(slots.router_pool(c, True, margin_supported=True)["memory_margin_mib"], 1536)
        c["slot_headroom_mib"] = 0
        self.assertNotIn("memory_margin_mib", slots.router_pool(c, True, margin_supported=True))


if __name__ == "__main__":
    unittest.main()
