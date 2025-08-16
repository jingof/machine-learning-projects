from bankai.data.synth_data import generate_all
from bankai.utils.io import ensure_dir
def test_generate(tmp_path):
    paths = {"data_dir": str(tmp_path/"data"), "artifacts_dir": str(tmp_path/"artifacts")}
    ensure_dir(paths["data_dir"]); ensure_dir(paths["artifacts_dir"])
    out = generate_all(paths, seed=1, n_customers=100, days=10)
    assert all(len(df)>0 for df in out.values())

