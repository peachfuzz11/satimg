from satproducts.deepzoom.dzi_factory import DZIFactory
from satproducts.products.landsat.landsat_product import LandsatProduct
from satproducts.products.sentinel.sentinel1.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct

if __name__ == "__main__":
    ps1 = ("/home/ph/git/satproducts/tests/data/LC09_L1TP_194022_20251013_20251013_02_T1", LandsatProduct)
    ps2 = (
        "/home/ph/git/satproducts/tests/data/S1C_IW_GRDH_1SDV_20251011T054028_20251011T054053_004510_008ED9_A87E_COG.SAFE",
        Sentinel1IWProduct)
    ps3 = ("/home/ph/git/satproducts/tests/data/S2B_MSIL1C_20251010T102849_N0511_R108_T33VUC_20251010T124345.SAFE",
           Sentinel2L1CProduct)
    for p, prod in (ps1,):# ps2, ps3):
        pro = prod(p)
        factory = DZIFactory(out_path=p)
        factory.create(pro)

    import http.server
    import socketserver

    PORT = 8000
    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()
