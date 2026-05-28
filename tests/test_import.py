"""Smoke tests for the MEGA-ODE package."""


def test_import_public_api():
    import megaode

    assert hasattr(megaode, "MEGAODE")
    assert hasattr(megaode, "load_demo_data")
    assert hasattr(megaode, "run_demo")


def test_load_demo_data_subset():
    from megaode import load_demo_data

    data = load_demo_data("demo_data", max_nodes=25)
    assert len(data) >= 1
    assert data.num_features == 1
    assert data[0].ndata["x12"].shape[0] == 25
