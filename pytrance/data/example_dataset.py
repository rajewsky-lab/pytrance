import pooch

FETCHER = pooch.create(
    path=pooch.os_cache("pyTrance"),
    base_url="",  
    registry={
        "u2os_merfish.h5ad": None,  
    },
    urls={
        "u2os_merfish.h5ad": "https://ndownloader.figshare.com/files/29046861",
    },
)

def load_example():
    import scanpy as sc
    fname = FETCHER.fetch("u2os_merfish.h5ad", progressbar=True)
    return sc.read_h5ad(fname)