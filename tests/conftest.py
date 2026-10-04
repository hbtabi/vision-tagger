import pytest

from visiontagger.samples import write_samples


@pytest.fixture(scope="session")
def sample_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("images")
    write_samples(d)
    return d
