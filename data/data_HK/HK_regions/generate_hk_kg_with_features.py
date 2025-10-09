import os
import json
import math
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.strtree import STRtree
from shapely import speedups

try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else []

try:
    import osmnx as ox
except Exception:
    ox = None

if speedups.available:
    speedups.enable()


def read_regions(geojson_path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(geojson_path)
    if "Station code" not in gdf.columns:
        raise ValueError("GeoJSON must contain 'Station code'.")
    keep_cols = [c for c in ["Station code",
                             "Station name", "geometry"] if c in gdf.columns]
    gdf = gdf[keep_cols].copy()
    gdf["region_id"] = gdf["Station code"].astype(str)
    gdf = gdf.set_index("region_id", drop=False)
    # Fix geometries
    gdf["geometry"] = gdf["geometry"].buffer(0)
    return gdf


def read_pois(poi_csv: str) -> pd.DataFrame:
    usecols = ["名称", "大类", "经度", "纬度"]
    df = pd.read_csv(poi_csv, usecols=[
                     c for c in usecols if c in pd.read_csv(poi_csv, nrows=0).columns])
    df = df.dropna(subset=["名称", "大类", "经度", "纬度"]).copy()
    return df


def build_spatial_index(geoms: gpd.GeoSeries) -> Tuple[STRtree, List, List]:
    geom_list = list(geoms.values)
    ids = list(geoms.index)
    tree = STRtree(geom_list)
    return tree, geom_list, ids


def assign_pois_to_regions(pois: pd.DataFrame, regions: gpd.GeoDataFrame) -> pd.DataFrame:
    tree, geom_list, id_list = build_spatial_index(regions["geometry"])
    centroids = regions["geometry"].centroid
    center_xy = np.array([[c.x, c.y] for c in centroids])
    region_ids = regions.index.to_list()

    assigned = []
    for _, r in tqdm(pois.iterrows(), total=len(pois), desc="Assign POIs", unit="poi"):
        pt = Point(float(r["经度"]), float(r["纬度"]))
        cand = tree.query(pt)
        found = None
        # STRtree.query returns candidate indices (shapely>=2)
        for idx in np.atleast_1d(cand):
            poly = geom_list[int(idx)]
            if poly.contains(pt):
                found = id_list[int(idx)]
                break
        if found is None:
            v = np.array([pt.x, pt.y])
            d2 = np.sum((center_xy - v) ** 2, axis=1)
            found = region_ids[int(np.argmin(d2))]
        assigned.append(found)
    out = pois.copy()
    out["region_id"] = assigned
    return out


def compute_region_graph(regions: gpd.GeoDataFrame, knn: int = 10) -> Tuple[set, set]:
    # BorderBy via touches with positive shared boundary length
    sindex = regions.sindex
    borders = set()
    for rid, geom in regions["geometry"].items():
        for j in sindex.intersection(geom.bounds):
            rid2 = regions.index[j]
            if rid2 == rid:
                continue
            geom2 = regions.loc[rid2, "geometry"]
            if geom.touches(geom2):
                inter = geom.boundary.intersection(geom2.boundary)
                if not inter.is_empty and inter.length > 0:
                    a, b = sorted([rid, rid2])
                    borders.add((a, b))
    # NearBy via KNN centroid Euclidean in lon/lat
    centers = regions["geometry"].centroid
    ids = regions.index.to_list()
    xy = np.array([[p.x, p.y] for p in centers])
    nearby = set()
    for i, rid in enumerate(ids):
        d2 = np.sum((xy - xy[i]) ** 2, axis=1)
        order = np.argsort(d2)
        for idx in order[1:knn+1]:
            a, b = sorted([rid, ids[idx]])
            nearby.add((a, b))
    return borders, nearby


def compute_road_density(regions: gpd.GeoDataFrame) -> Dict[str, float]:
    if ox is None:
        # Fallback: zeros if osmnx unavailable
        return {rid: 0.0 for rid in regions.index}
    road_density = {}
    for rid, row in tqdm(regions.iterrows(), total=len(regions), desc="Road density", unit="reg"):
        poly = row["geometry"]
        try:
            G = ox.graph_from_polygon(
                poly, network_type="drive", simplify=True)
            edges = ox.graph_to_gdfs(G, nodes=False)
            length_m = edges["length"].sum(
            ) if "length" in edges.columns else edges.geometry.length.sum() * 111_000
            area_deg2 = poly.area
            area_m2 = area_deg2 * (111_000 ** 2)  # rough conversion
            rd_km_per_km2 = (length_m / 1000.0) / \
                max(area_m2 / 1_000_000.0, 1e-6)
        except Exception:
            rd_km_per_km2 = 0.0
        road_density[rid] = float(rd_km_per_km2)
    return road_density


def compute_function_bias(pois_assigned: pd.DataFrame) -> Dict[str, float]:
    # Map 大类到三大功能：commercial, residential, tourism
    cat_to_func = {
        "公司企业": "commercial",
        "购物消费": "commercial",
        "餐饮美食": "commercial",
        "商务住宅": "residential",
        "生活服务": "residential",
        "旅游景点": "tourism",
        "科教文化": "residential",
        "交通设施": "commercial",
        "运动健身": "residential",
        "医疗": "residential",
    }
    counts = {}
    for rid, g in pois_assigned.groupby("region_id"):
        func_counts = {"commercial": 0, "residential": 0, "tourism": 0}
        for c, n in g["大类"].value_counts().items():
            f = cat_to_func.get(str(c), "residential")
            func_counts[f] += int(n)
        total = sum(func_counts.values())
        if total == 0:
            bias = 0.0
        else:
            comm_share = func_counts["commercial"]/total
            resid_share = func_counts["residential"]/total
            # bias: commercial vs residential（旅游不计入对比，作为噪声）
            bias = float(comm_share - resid_share)
        counts[rid] = bias
    return counts


def zscore(series: pd.Series) -> pd.Series:
    x = series.astype(float)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma == 0 or np.isnan(sigma):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sigma


def build_features(regions: gpd.GeoDataFrame,
                   pois_assigned: pd.DataFrame,
                   road_density: Dict[str, float]) -> Tuple[Dict, Dict[str, List[float]]]:
    # Base metrics per region
    n_pois = pois_assigned.groupby("region_id").size().reindex(
        regions.index, fill_value=0)
    area_deg2 = regions["geometry"].area
    area_km2 = area_deg2 * (111_000**2) / 1_000_000

    # Proxies
    # pop: POI 密度（每 km2）
    pop_proxy = (n_pois / area_km2.replace(0, np.nan)).fillna(0)
    # edu: 教育相关 POI 每 km2
    edu_mask = pois_assigned["大类"].astype(str).eq("科教文化")
    edu_counts = pois_assigned.loc[edu_mask].groupby(
        "region_id").size().reindex(regions.index, fill_value=0)
    edu_proxy = (edu_counts / area_km2.replace(0, np.nan)).fillna(0)
    # income: 商业相关 POI 占比（商业/(商业+居住)）
    comm_mask = pois_assigned["大类"].isin(["公司企业", "购物消费", "餐饮美食", "交通设施"])
    resid_mask = pois_assigned["大类"].isin(
        ["商务住宅", "生活服务", "科教文化", "医疗", "运动健身"])
    comm = pois_assigned.loc[comm_mask].groupby(
        "region_id").size().reindex(regions.index, fill_value=0)
    resid = pois_assigned.loc[resid_mask].groupby(
        "region_id").size().reindex(regions.index, fill_value=0)
    denom = (comm + resid).replace(0, np.nan)
    income_proxy = (comm / denom).fillna(0.0)
    # road density already in km/km2
    road_series = pd.Series(road_density).reindex(regions.index).fillna(0.0)
    # functional bias scalar
    func_bias = pd.Series(compute_function_bias(
        pois_assigned)).reindex(regions.index).fillna(0.0)

    # Z-score to form 6D feature: [z(n_pois), z(pop), z(edu), z(income), z(road_density), z(func_bias)]
    feat_df = pd.DataFrame({
        "n_pois": n_pois,
        "pop_proxy": pop_proxy,
        "edu_proxy": edu_proxy,
        "income_proxy": income_proxy,
        "road_density": road_series,
        "func_bias": func_bias,
    })

    z_df = feat_df.apply(zscore)
    feature_map = {rid: z_df.loc[rid].tolist() for rid in z_df.index}

    # Pack human-readable metrics
    metrics = {
        rid: {
            "n_pois": int(feat_df.loc[rid, "n_pois"]),
            "pop": float(feat_df.loc[rid, "pop_proxy"]),
            "edu": float(feat_df.loc[rid, "edu_proxy"]),
            "income": float(feat_df.loc[rid, "income_proxy"]),
            "road_density": float(feat_df.loc[rid, "road_density"]),
            "func_bias": float(feat_df.loc[rid, "func_bias"]),
        }
        for rid in feat_df.index
    }
    return metrics, feature_map


def write_outputs(out_dir: str,
                  regions: gpd.GeoDataFrame,
                  pois_assigned: pd.DataFrame,
                  border_pairs: set,
                  nearby_pairs: set,
                  feature_metrics: Dict[str, Dict],
                  features: Dict[str, List[float]]):
    os.makedirs(out_dir, exist_ok=True)
    # kg.txt
    kg_path = os.path.join(out_dir, "kg.txt")
    with open(kg_path, "w", encoding="utf-8") as f:
        # region-region
        for a, b in sorted(border_pairs):
            f.write(f"{a}\tBorderBy\t{b}\n")
            f.write(f"{b}\tBorderBy\t{a}\n")
        for a, b in sorted(nearby_pairs):
            f.write(f"{a}\tNearBy\t{b}\n")
            f.write(f"{b}\tNearBy\t{a}\n")
        # POI relations
        for _, r in tqdm(pois_assigned.iterrows(), total=len(pois_assigned), desc="Write POI triples", unit="poi"):
            pid = f"poi:{r['名称']}|{round(float(r['经度']),3)}_{round(float(r['纬度']),3)}"
            cat = f"cat:{r['大类']}"
            f.write(f"{pid}\tcateOf\t{cat}\n")
            f.write(f"{pid}\tLocateAt\t{r['region_id']}\n")

    # region2info.json (NYC-like)
    info = {}
    for rid, row in regions.iterrows():
        c = row["geometry"].centroid
        m = feature_metrics[rid]
        info[rid] = {
            "center": [float(c.x), float(c.y)],
            "n_pois": m["n_pois"],
            "pop": m["pop"],
            "edu": m["edu"],
            "income": m["income"],
            "road_density": m["road_density"],
            "func_bias": m["func_bias"],
            "feature": features[rid],
        }
    with open(os.path.join(out_dir, "region2info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False)

    return kg_path


def main():
    parser = argparse.ArgumentParser("Generate HK KG and 6D features")
    parser.add_argument("--geojson", required=True)
    parser.add_argument("--pois", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--knn", type=int, default=10)
    args = parser.parse_args()

    regions = read_regions(args.geojson)
    pois = read_pois(args.pois)
    pois_assigned = assign_pois_to_regions(pois, regions)

    border_pairs, nearby_pairs = compute_region_graph(regions, knn=args.knn)
    road_density = compute_road_density(regions)
    metrics, feature_map = build_features(regions, pois_assigned, road_density)

    kg_path = write_outputs(args.out_dir, regions, pois_assigned,
                            border_pairs, nearby_pairs, metrics, feature_map)
    print("Saved:", kg_path)
    print("Saved:", os.path.join(args.out_dir, "region2info.json"))
    print("Regions:", len(regions))
    print("POIs assigned:", len(pois_assigned))


if __name__ == "__main__":
    main()
