#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BƯỚC 2 — TIỀN XỬ LÝ LIDC-IDRI → CACHE .npz CHO NOTEBOOK train_lung_cancer_lidc

Đọc DICOM + XML chú thích của 4 bác sĩ, với MỖI vùng khoanh <roi> của MỖI bác sĩ:
  1. Đọc lát cắt DICOM tương ứng (theo imageSOP_UID) → đổi sang Hounsfield Unit (HU).
  2. Vẽ mask khối u 512x512 từ đa giác <edgeMap> (trừ các vùng inclusion=FALSE).
  3. Lấy tâm khối u (cx, cy), cắt patch 64x64 quanh tâm.
  4. Ghép 3 cửa sổ HU (R: Phổi, G: Trung thất, B: Xương) → ảnh uint8 64x64x3.
  5. Lưu kèm điểm ác tính, 8 thuộc tính vi thể, mã bệnh nhân, toạ độ x, y, z.

Kết quả: tệp .npz với các khoá
  patches (N,64,64,3) uint8 | masks (N,64,64) uint8 | labels (N,) int64 | malignancies (N,) float32
  pids (N,) str | attrs (N,8) float32 | cxs, cys, zs (N,) float32
  + (phụ) nodule_ids, readers, sop_uids để truy vết.
Đồng thời ghi metadata/xml_series_index.csv (dùng cho phần minh hoạ ở Mục 1 của notebook).

Cách dùng (chạy trong Command Prompt / Terminal, KHÔNG nên chạy multiprocessing trong Jupyter trên Windows):
  python preprocess_lidc.py --data_dir lidc_idri --out lidc_nodules_2d_100pct_cache.npz
  python preprocess_lidc.py --data_dir lidc_idri --out smoke_cache.npz --limit 10      # bản nhỏ để chạy thử
  python preprocess_lidc.py --xml_dir tcia-lidc-xml     # nếu XML chú thích được tải riêng (gói LIDC-XML-only)
Trong Jupyter có thể gọi:  !python preprocess_lidc.py --data_dir lidc_idri

Cấu trúc thư mục mong đợi (đúng như khi tải từ TCIA bằng NBIA Data Retriever, kiểu "Descriptive Directory Name"):
  lidc_idri/
    LIDC-IDRI-0001/<study>/<series>/*.dcm  và  <series>/*.xml
    LIDC-IDRI-0002/...
Script tự tìm *.xml và *.dcm đệ quy trong từng thư mục bệnh nhân nên tên thư mục con không quan trọng.
Nếu XML nằm riêng (vd. tcia-lidc-xml/185/069.xml), script tự quét cạnh script và ghép theo SeriesInstanceUID.
DICOM không có đuôi .dcm cũng được nhận diện (theo chữ ký 'DICM').

Thư viện: numpy, pandas, opencv-python, pydicom (pip install numpy pandas opencv-python pydicom)
"""
import argparse
import glob
import os
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import numpy as np
import pandas as pd
import pydicom

# ----------------------------------------------------------------------------
# Hằng số (khớp với notebook)
# ----------------------------------------------------------------------------
PATCH = 64
IMG_SIZE = 512
AIR_HU = -1024.0
ATTR_NAMES = ['subtlety', 'internalStructure', 'calcification', 'sphericity',
              'margin', 'lobulation', 'spiculation', 'texture']
# Cửa sổ HU giống hàm get_*_window trong Mục 1 của notebook: (center, width)
WINDOWS = [(-600, 1500),   # R: Phổi        [-1350, +150]
           (40, 350),      # G: Trung thất  [-135, +215]
           (300, 1000)]    # B: Xương/Vôi   [-200, +800]


# ----------------------------------------------------------------------------
# Tiện ích XML (bỏ namespace http://www.nih.gov)
# ----------------------------------------------------------------------------
def _local(tag):
    return tag.split('}', 1)[-1]


def _children(el, name):
    return [c for c in el if _local(c.tag) == name]


def _child(el, name):
    for c in el:
        if _local(c.tag) == name:
            return c
    return None


def _text(el, name, default=None):
    c = _child(el, name)
    return c.text.strip() if c is not None and c.text is not None else default


def parse_lidc_xml(xml_path):
    """Trả về (series_uid, list các bản ghi ROI). Bỏ qua XML không phải CT (IdriReadMessage của X-quang)."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None, []
    if _local(root.tag) != 'LidcReadMessage':
        return None, []
    header = _child(root, 'ResponseHeader')
    series_uid = _text(header, 'SeriesInstanceUid') if header is not None else None

    records = []
    for r_idx, session in enumerate(_children(root, 'readingSession')):
        reader = _text(session, 'servicingRadiologistID', f'reader{r_idx}')
        for nod in _children(session, 'unblindedReadNodule'):
            ch = _child(nod, 'characteristics')
            if ch is None or _text(ch, 'malignancy') is None:
                continue   # nốt < 3 mm không có thuộc tính → bỏ (giống quy ước LIDC)
            try:
                mal = float(_text(ch, 'malignancy'))
                attrs = [float(_text(ch, a, 'nan')) for a in ATTR_NAMES]
            except ValueError:
                continue
            nod_id = _text(nod, 'noduleID', '?')
            for roi in _children(nod, 'roi'):
                sop = _text(roi, 'imageSOP_UID')
                z = float(_text(roi, 'imageZposition', 'nan'))
                incl = (_text(roi, 'inclusion', 'TRUE').upper() == 'TRUE')
                pts = [(int(float(_text(e, 'xCoord'))), int(float(_text(e, 'yCoord'))))
                       for e in _children(roi, 'edgeMap')]
                if sop and len(pts) >= 3:
                    records.append(dict(reader=reader, nodule_id=nod_id, sop=sop, z=z, inclusion=incl,
                                        pts=np.array(pts, np.int32), mal=mal, attrs=attrs))
    return series_uid, records


# ----------------------------------------------------------------------------
# Tiện ích DICOM / ảnh
# ----------------------------------------------------------------------------
def to_hu(ds):
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    inter = float(getattr(ds, 'RescaleIntercept', -1024.0))
    hu = img * slope + inter
    return np.clip(hu, AIR_HU, 3071.0)   # vùng padding ngoài trường nhìn (-2000/-3024) → khí


def tri_window_uint8(hu):
    chans = []
    for c, w in WINDOWS:
        lo, hi = c - w / 2.0, c + w / 2.0
        chans.append(np.clip((hu - lo) / (hi - lo), 0.0, 1.0))
    return (np.stack(chans, axis=-1) * 255.0 + 0.5).astype(np.uint8)


def crop_center(arr, cx, cy, size, fill):
    """Cắt vùng size x size quanh (cx, cy), đệm `fill` nếu tràn biên ảnh."""
    half = size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    pad = size
    if arr.ndim == 2:
        padded = np.pad(arr, pad, mode='constant', constant_values=fill)
    else:
        padded = np.pad(arr, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=fill)
    return padded[y0 + pad:y0 + pad + size, x0 + pad:x0 + pad + size]


def index_dicoms(dcm_paths):
    """SOPInstanceUID → (đường dẫn, SeriesInstanceUID, Modality), chỉ đọc header."""
    idx = {}
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True,
                                 specific_tags=['SOPInstanceUID', 'SeriesInstanceUID', 'Modality'])
            idx[str(ds.SOPInstanceUID)] = (p, str(getattr(ds, 'SeriesInstanceUID', '')), str(getattr(ds, 'Modality', '')))
        except Exception:
            continue
    return idx


# ----------------------------------------------------------------------------
# Xử lý 1 bệnh nhân
# ----------------------------------------------------------------------------
def _is_dicom_file(path):
    """Nhận diện DICOM theo chữ ký 'DICM' ở byte 128 (dùng cho tệp không có đuôi .dcm)."""
    try:
        with open(path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except OSError:
        return False


def find_dicoms(patient_dir):
    files = [p for p in glob.glob(os.path.join(patient_dir, '**', '*'), recursive=True) if os.path.isfile(p)]
    dcms = [p for p in files if p.lower().endswith('.dcm')]
    if not dcms:   # một số công cụ tải về lưu DICOM không có đuôi
        skip = ('.xml', '.csv', '.txt', '.json', '.zip', '.jpg', '.png', '.npz', '.py')
        dcms = [p for p in files if not p.lower().endswith(skip) and _is_dicom_file(p)]
    return dcms, files


def process_patient(patient_dir, xml_by_series=None):
    pid = os.path.basename(os.path.normpath(patient_dir))
    dcms, files = find_dicoms(patient_dir)
    xmls = [p for p in files if p.lower().endswith('.xml')]
    out = dict(patches=[], masks=[], mal=[], attrs=[], cx=[], cy=[], z=[], nod=[], reader=[], sop=[])
    index_rows = []
    status = dict(pid=pid, n_files=len(files), n_dcm=len(dcms), n_xml_local=len(xmls), n_xml_ext=0, n_roi=0,
                  n_slices=0, ext_sample=sorted({os.path.splitext(p)[1].lower() or '(không đuôi)' for p in files})[:6])
    warnings = []
    if not dcms:
        warnings.append(f'{pid}: không có tệp DICOM ({len(files)} tệp trong thư mục)')
        return pid, out, index_rows, warnings, status

    sop_index = index_dicoms(dcms)
    series_count = pd.Series([v[1] for v in sop_index.values()], dtype=object).value_counts().to_dict()
    if xml_by_series:   # XML để ở thư mục riêng → ghép theo SeriesInstanceUID
        local_series = {_series_uid_of_xml(x) for x in xmls}   # tránh xử lý trùng nếu XML có ở cả 2 nơi
        ext = [xml_by_series[s] for s in series_count if s in xml_by_series and s not in local_series]
        status['n_xml_ext'] = len(ext)
        xmls = xmls + [p for p in ext if p not in xmls]
    if not xmls:
        warnings.append(f'{pid}: không tìm thấy XML chú thích cho series nào')
        return pid, out, index_rows, warnings, status
    hu_cache = {}

    for xml_path in xmls:
        series_uid, records = parse_lidc_xml(xml_path)
        if series_uid is None:
            continue
        modality = next((m for (_, s, m) in sop_index.values() if s == series_uid), 'NA')
        index_rows.append(dict(PatientID=pid, Modality=modality, SeriesInstanceUID=series_uid,
                               xml_path=os.path.relpath(xml_path), n_dcm=series_count.get(series_uid, 0),
                               n_roi=len(records)))
        if modality != 'CT':
            continue
        status['n_roi'] += len(records)

        # Gom các đa giác theo (bác sĩ, nốt, lát) để trừ vùng inclusion=FALSE
        groups = {}
        for r in records:
            groups.setdefault((r['reader'], r['nodule_id'], r['sop']), []).append(r)

        for (reader, nod_id, sop), rs in groups.items():
            if sop not in sop_index:
                warnings.append(f'{pid}: không tìm thấy DICOM cho SOP {sop[-12:]}')
                continue
            if sop not in hu_cache:
                ds = pydicom.dcmread(sop_index[sop][0])
                hu = to_hu(ds)
                scale = 1.0
                if hu.shape != (IMG_SIZE, IMG_SIZE):
                    scale = IMG_SIZE / float(hu.shape[1])
                    hu = cv2.resize(hu, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                hu_cache[sop] = (hu, scale)
            hu, scale = hu_cache[sop]

            mask = np.zeros((IMG_SIZE, IMG_SIZE), np.uint8)
            for r in rs:
                if r['inclusion']:
                    cv2.fillPoly(mask, [np.round(r['pts'] * scale).astype(np.int32)], 1)
            for r in rs:
                if not r['inclusion']:
                    cv2.fillPoly(mask, [np.round(r['pts'] * scale).astype(np.int32)], 0)
            if mask.sum() == 0:
                continue

            ys, xs = np.nonzero(mask)
            cx, cy = float(xs.mean()), float(ys.mean())
            patch = tri_window_uint8(crop_center(hu, cx, cy, PATCH, AIR_HU))
            m64 = crop_center(mask, cx, cy, PATCH, 0)
            r0 = rs[0]
            out['patches'].append(patch); out['masks'].append(m64)
            out['mal'].append(r0['mal']); out['attrs'].append(r0['attrs'])
            out['cx'].append(cx); out['cy'].append(cy); out['z'].append(r0['z'])
            out['nod'].append(str(nod_id)); out['reader'].append(str(reader)); out['sop'].append(sop)
            status['n_slices'] += 1

        if len(hu_cache) > 400:   # giới hạn RAM
            hu_cache.clear()
    return pid, out, index_rows, warnings, status


def _series_uid_of_xml(path):
    """Đọc nhanh SeriesInstanceUid trong phần ResponseHeader của XML LIDC (None nếu không phải XML CT)."""
    try:
        for _, el in ET.iterparse(path, events=('end',)):
            name = _local(el.tag)
            if name == 'SeriesInstanceUid' and el.text:
                return el.text.strip()
            if name in ('readingSession', 'IdriReadMessage'):
                return None
    except ET.ParseError:
        return None
    return None


def build_xml_map(roots, max_depth=6):
    """Tìm các XML nằm NGOÀI thư mục bệnh nhân (ví dụ gói LIDC-XML-only: tcia-lidc-xml/185/069.xml)
    và lập bảng SeriesInstanceUID → đường dẫn XML."""
    xml_map, n = {}, 0
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        base_depth = root.rstrip(os.sep).count(os.sep)
        for cur, dirs, fnames in os.walk(root):
            if cur.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
            dirs[:] = [d for d in dirs if not d.upper().startswith('LIDC-IDRI-') and not d.startswith('.')]
            for f in fnames:
                if f.lower().endswith('.xml'):
                    n += 1
                    uid = _series_uid_of_xml(os.path.join(cur, f))
                    if uid:
                        xml_map.setdefault(uid, os.path.join(cur, f))
    return xml_map, n


# ----------------------------------------------------------------------------
# Tìm thư mục dữ liệu
# ----------------------------------------------------------------------------
def _has_patients(d):
    return os.path.isdir(d) and any(n.upper().startswith('LIDC-IDRI-') and os.path.isdir(os.path.join(d, n))
                                    for n in os.listdir(d))


def find_data_dir(given, script_dir, max_depth=4):
    """Trả về thư mục chứa trực tiếp các thư mục LIDC-IDRI-xxxx.
    Thử: đường dẫn được chỉ định → các thư mục con của nó → các thư mục con của thư mục script."""
    cands = [given] if os.path.isabs(given) else [os.path.join(script_dir, given), os.path.abspath(given)]
    for c in cands:
        if _has_patients(c):
            return os.path.abspath(c)
    roots = [c for c in cands if os.path.isdir(c)] + [script_dir]
    for root in roots:
        base_depth = root.rstrip(os.sep).count(os.sep)
        for cur, dirs, _ in os.walk(root):
            if cur.rstrip(os.sep).count(os.sep) - base_depth >= max_depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if not d.upper().startswith('LIDC-IDRI-') and not d.startswith('.')]
            if _has_patients(cur):
                return os.path.abspath(cur)
    return None


def print_diagnostics(all_status, all_warn):
    st = pd.DataFrame(all_status)
    if st.empty:
        return
    print('\n' + '-' * 72 + '\n🔎 CHẨN ĐOÁN')
    print(f"   Bệnh nhân có DICOM: {(st.n_dcm > 0).sum()}/{len(st)} | tổng số tệp DICOM: {st.n_dcm.sum():,}")
    print(f"   Bệnh nhân có XML (trong thư mục / ghép từ thư mục riêng): "
          f"{(st.n_xml_local > 0).sum()} / {(st.n_xml_ext > 0).sum()}")
    print(f"   Bệnh nhân có ROI CT: {(st.n_roi > 0).sum()} | trích được lát: {(st.n_slices > 0).sum()}")
    if (st.n_dcm == 0).all():
        ex = st.sort_values(['n_files', 'pid'], ascending=[False, True]).iloc[0]
        print(f"   ⚠️ Không thấy DICOM nào. Ví dụ {ex.pid}: {ex.n_files} tệp, đuôi tệp: {ex.ext_sample}")
        print("      → Thư mục bệnh nhân có thể rỗng / chỉ có tệp nén .zip / dữ liệu chưa tải xong.")
    elif (st.n_xml_local + st.n_xml_ext == 0).all():
        print("   ⚠️ Có DICOM nhưng KHÔNG có XML chú thích nào.")
        print("      → Tải gói 'LIDC-XML-only' (Radiologist Annotations/Segmentations) trên trang TCIA của LIDC-IDRI,")
        print("        giải nén cạnh script rồi chạy lại (hoặc chỉ rõ bằng --xml_dir \"D:\\tcia-lidc-xml\").")
    if all_warn:
        print(f'   {len(all_warn)} cảnh báo, ví dụ:', *all_warn[:5], sep='\n     ')
    print('-' * 72)


# ----------------------------------------------------------------------------
# Chương trình chính
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Tiền xử lý LIDC-IDRI → cache .npz')
    ap.add_argument('--data_dir', default='lidc_idri', help='thư mục chứa các thư mục LIDC-IDRI-xxxx')
    ap.add_argument('--out', default='lidc_nodules_2d_100pct_cache.npz')
    ap.add_argument('--index_csv', default=None,
                    help="mặc định metadata/xml_series_index.csv; khi dùng --limit thì metadata/xml_series_index_<tên out>.csv")
    ap.add_argument('--limit', type=int, default=0, help='chỉ xử lý N bệnh nhân đầu (0 = tất cả)')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--xml_dir', default=None,
                    help='thư mục chứa XML chú thích nếu XML KHÔNG nằm cùng thư mục DICOM (vd. tcia-lidc-xml)')
    args = ap.parse_args()

    # Đường dẫn tương đối được tính từ THƯ MỤC CHỨA SCRIPT (không phụ thuộc terminal đang đứng ở đâu),
    # nên cache .npz và metadata/ luôn được ghi cạnh script / notebook.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f'📂 Thư mục làm việc: {script_dir}')

    data_dir = find_data_dir(args.data_dir, script_dir)
    if data_dir is None:
        sys.exit(f"\n❌ Không tìm thấy dữ liệu LIDC-IDRI.\n"
                 f"   Đã tìm: {os.path.abspath(args.data_dir)}\n"
                 f"   và các thư mục con (tối đa 4 cấp) của {script_dir} có chứa thư mục LIDC-IDRI-xxxx.\n"
                 f"   Các mục đang có trong {script_dir}: {sorted(os.listdir(script_dir))[:20]}\n"
                 f"   → Chỉ rõ đường dẫn, ví dụ:  python preprocess_lidc.py --data_dir \"D:\\LIDC\\manifest-xxxx\\LIDC-IDRI\"")
    print(f'🗂️  Dữ liệu: {data_dir}')

    if args.index_csv is None:
        stem = os.path.splitext(os.path.basename(args.out))[0]
        args.index_csv = os.path.join('metadata', 'xml_series_index.csv' if not args.limit else f'xml_series_index_{stem}.csv')

    patient_dirs = sorted(d for d in glob.glob(os.path.join(data_dir, '*')) if os.path.isdir(d))
    if args.limit:
        patient_dirs = patient_dirs[:args.limit]

    # XML nằm ngoài thư mục bệnh nhân? (gói LIDC-XML-only tải riêng từ TCIA)
    roots = [os.path.abspath(args.xml_dir)] if args.xml_dir else [script_dir, os.path.dirname(data_dir)]
    xml_map, n_ext = build_xml_map(list(dict.fromkeys(roots)))
    if args.xml_dir and not xml_map:
        sys.exit(f'❌ Không tìm thấy XML chú thích CT nào trong --xml_dir {args.xml_dir}')
    print(f'📝 XML ngoài thư mục bệnh nhân: {len(xml_map)} series CT (đã quét {n_ext} tệp .xml)')
    print(f'🩻 Xử lý {len(patient_dirs)} bệnh nhân với {args.workers} tiến trình...')

    t0 = time.time()
    results, all_index, all_warn, all_status = {}, [], [], []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_patient, d, xml_map): d for d in patient_dirs}
            for k, f in enumerate(as_completed(futs), 1):
                pid, out, idx_rows, warn, st = f.result()
                results[pid] = out; all_index += idx_rows; all_warn += warn; all_status.append(st)
                if k % 20 == 0 or k == len(futs):
                    print(f'   {k}/{len(futs)} bệnh nhân | {time.time()-t0:.0f}s')
    else:
        for k, d in enumerate(patient_dirs, 1):
            pid, out, idx_rows, warn, st = process_patient(d, xml_map)
            results[pid] = out; all_index += idx_rows; all_warn += warn; all_status.append(st)
            if k % 20 == 0 or k == len(patient_dirs):
                print(f'   {k}/{len(patient_dirs)} bệnh nhân | {time.time()-t0:.0f}s')

    # Ghép theo thứ tự mã bệnh nhân để kết quả tái lập được
    keys = ['patches', 'masks', 'mal', 'attrs', 'cx', 'cy', 'z', 'nod', 'reader', 'sop']
    cat = {k: [] for k in keys}; pids = []
    for pid in sorted(results):
        o = results[pid]
        for k in keys:
            cat[k] += o[k]
        pids += [pid] * len(o['mal'])
    print_diagnostics(all_status, all_warn)
    if not pids:
        sys.exit('\n❌ Không trích xuất được lát cắt nào. Xem phần CHẨN ĐOÁN phía trên.')

    mal = np.array(cat['mal'], np.float32)
    labels = np.where(mal > 3, 1, np.where(mal < 3, 0, -1)).astype(np.int64)   # 3 = mơ hồ
    data = dict(
        patches=np.stack(cat['patches']).astype(np.uint8),
        masks=np.stack(cat['masks']).astype(np.uint8),
        labels=labels,
        malignancies=mal,
        pids=np.array(pids, dtype=str),                       # kiểu chuỗi, không cần allow_pickle
        attrs=np.array(cat['attrs'], np.float32),
        cxs=np.array(cat['cx'], np.float32),
        cys=np.array(cat['cy'], np.float32),
        zs=np.array(cat['z'], np.float32),
        nodule_ids=np.array(cat['nod'], dtype=str),
        readers=np.array(cat['reader'], dtype=str),
        sop_uids=np.array(cat['sop'], dtype=str),
    )
    np.savez_compressed(args.out, **data)

    os.makedirs(os.path.dirname(args.index_csv) or '.', exist_ok=True)
    pd.DataFrame(all_index).to_csv(args.index_csv, index=False)

    n = len(labels)
    print('\n' + '=' * 72)
    print(f'✅ Đã lưu {args.out} ({os.path.getsize(args.out)/1024**2:.2f} MB) sau {time.time()-t0:.0f}s')
    print(f'   Lát cắt (bản ghi ROI): {n:,} | Bệnh nhân có nốt: {len(np.unique(data["pids"])):,}')
    print(f'   Nhãn lát: lành(1–2)={np.sum(labels==0):,} | ác(4–5)={np.sum(labels==1):,} | mơ hồ(3)={np.sum(labels==-1):,}')
    print(f'   Diện tích mask: trung vị {np.median(data["masks"].reshape(n,-1).sum(1)):.0f} px | '
          f'mask chạm biên patch: {int((data["masks"][:, [0,-1], :].any((1,2)) | data["masks"][:, :, [0,-1]].any((1,2))).sum())}')
    print(f'   Chỉ mục series: {args.index_csv} ({len(all_index)} dòng)')


if __name__ == '__main__':
    main()
