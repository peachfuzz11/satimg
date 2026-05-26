from satproducts.deepzoom.dzi_factory import DZIFactory
from satproducts.products.landsat.landsat_product import LandsatProduct
from satproducts.products.sentinel.sentinel1.iw.sentinel1_iw_product import Sentinel1IWProduct
from satproducts.products.sentinel.sentinel2.sentinel2_l1c_product import Sentinel2L1CProduct

if __name__ == "__main__":
    ps1 = ("/home/ph/git/satproducts/tests/data/LC09_L1TP_194022_20251013_20251013_02_T1", LandsatProduct)
    ps2 = (
        "/home/ph/git/satproducts/tests/data/products_minified/S1A_IW_GRDH_1SDH_20141006T074149_20141006T074214_002706_00306E_51E0_COG.SAFE",
        Sentinel1IWProduct)
    ps3 = ("/home/ph/git/satproducts/tests/data/S2B_MSIL1C_20251010T102849_N0511_R108_T33VUC_20251010T124345.SAFE",
           Sentinel2L1CProduct)
    for p, prod in (ps2,):# ps2, ps3):
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
