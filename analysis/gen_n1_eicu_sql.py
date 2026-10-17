# -*- coding: utf-8 -*-
"""gen_n1_eicu_sql.py —— 生成 eICU 侧 N1 提取 SQL（与 MIMIC 短名对齐，供外部验证）"""
import os, csv
OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

# 1) 导出 (patientunitstayid, t0) 供 \copy
src = r"D:/SEEDANCE/EICU/csv/eicu_s3.csv"
coh = os.path.join(OUT, "_eicu_coh.csv")
n = 0
with open(src, encoding="utf-8", errors="replace", newline="") as f, \
     open(coh, "w", encoding="utf-8", newline="") as g:
    r = csv.DictReader(f); w = csv.writer(g); w.writerow(["patientunitstayid", "t0"])
    for row in r:
        if (row.get("cr_base24") or "").strip() in ("", "NA", "NaN"):
            continue
        w.writerow([row["patientunitstayid"], row["t0"]]); n += 1
print("cohort rows:", n)

LABS = [("BUN", "bun"), ("creatinine", "cr"), ("WBC x 1000", "wbc"), ("Hct", "hct"),
        ("Hgb", "hb"), ("platelets x 1000", "plt"), ("RDW", "rdw"), ("MCV", "mcv"),
        ("MCH", "mch"), ("MCHC", "mchc"), ("sodium", "na"), ("potassium", "k"),
        ("chloride", "cl"), ("calcium", "ca"), ("bicarbonate", "hco3"), ("anion gap", "agap"),
        ("glucose", "glu"), ("lactate", "lac"), ("pH", "ph"), ("paO2", "po2"),
        ("paCO2", "pco2"), ("FiO2", "fio2"), ("albumin", "alb"), ("total protein", "tp")]
VITS_AP = [("noninvasivesystolic", "sbpn"), ("noninvasivediastolic", "dbpn"),
           ("noninvasivemean", "mbpn")]
VITS = [("temperature", "temp"), ("sao2", "spo2"), ("heartrate", "hr"),
        ("respiration", "rr"), ("systemicsystolic", "sbp"),
        ("systemicdiastolic", "dbp"), ("systemicmean", "mbp")]

def lab_block(tag):
    cond = ("l.labresultoffset > s.t0 and l.labresultoffset <= s.t0 + 1440" if tag == "post"
            else "l.labresultoffset > s.t0 - 1440 and l.labresultoffset <= s.t0")
    lines = ["  min(case when l.labname = '%s' then l.labresult end) as %s_%s_min,"
             % (ln, sn, tag) for ln, sn in LABS]
    lines += ["  max(case when l.labname = '%s' then l.labresult end) as %s_%s_max,"
              % (ln, sn, tag) for ln, sn in LABS]
    lines.append("  count(*) as n_lab_%s" % tag)
    return ("drop table if exists lab_%s;\ncreate temp table lab_%s as\n"
            "select s.patientunitstayid,\n%s\nfrom coh2 s join lab l on l.patientunitstayid = s.patientunitstayid\n"
            "  and %s\n  and l.labname in (%s)\ngroup by s.patientunitstayid;\n"
            % (tag, tag, "\n".join(lines), cond,
               ",".join("'%s'" % ln for ln, _ in LABS)))

def vit_block(tag):
    cond = ("v.observationoffset > s.t0 and v.observationoffset <= s.t0 + 1440" if tag == "post"
            else "v.observationoffset > s.t0 - 1440 and v.observationoffset <= s.t0")
    lines = ["  min(v.%s) as %s_%s_min, max(v.%s) as %s_%s_max," % (c, s, tag, c, s, tag)
             for c, s in VITS]
    lines.append("  count(*) as n_vit_%s" % tag)
    return ("drop table if exists vit_%s;\ncreate temp table vit_%s as\n"
            "select s.patientunitstayid,\n%s\nfrom coh2 s join vitalperiodic v on v.patientunitstayid = s.patientunitstayid\n"
            "  and %s\ngroup by s.patientunitstayid;\n" % (tag, tag, "\n".join(lines), cond))

def vitap_block(tag):
    cond = ("v.observationoffset > s.t0 and v.observationoffset <= s.t0 + 1440" if tag == "post"
            else "v.observationoffset > s.t0 - 1440 and v.observationoffset <= s.t0")
    lines = ["  min(v.%s) as %s_%s_min, max(v.%s) as %s_%s_max," % (c, sn, tag, c, sn, tag)
             for c, sn in VITS_AP]
    lines.append("  count(*) as n_vitap_%s" % tag)
    return ("drop table if exists vitap_%s;\ncreate temp table vitap_%s as\n"
            "select s.patientunitstayid,\n%s\nfrom coh2 s join vitalaperiodic v on v.patientunitstayid = s.patientunitstayid\n"
            "  and %s\ngroup by s.patientunitstayid;\n" % (tag, tag, "\n".join(lines), cond))

picks = ["s.patientunitstayid"]
for tag in ("pre", "post"):
    for _, sn in LABS:
        picks += ["  a_%s.%s_%s_min" % (tag, sn, tag), "  a_%s.%s_%s_max" % (tag, sn, tag)]
    picks.append("  a_%s.n_lab_%s" % (tag, tag))
    for _, sn in VITS:
        picks += ["  b_%s.%s_%s_min" % (tag, sn, tag), "  b_%s.%s_%s_max" % (tag, sn, tag)]
    picks.append("  b_%s.n_vit_%s" % (tag, tag))
    for _, sn in VITS_AP:
        picks += ["  c_%s.%s_%s_min" % (tag, sn, tag), "  c_%s.%s_%s_max" % (tag, sn, tag)]
    picks.append("  c_%s.n_vitap_%s" % (tag, tag))

final = ["drop table if exists n1_eicu_labs;", "create temp table n1_eicu_labs as",
         "select", ",\n".join(picks), "from coh2 s",
         " left join lab_pre a_pre on a_pre.patientunitstayid = s.patientunitstayid",
         " left join lab_post a_post on a_post.patientunitstayid = s.patientunitstayid",
         " left join vit_pre b_pre on b_pre.patientunitstayid = s.patientunitstayid",
         " left join vit_post b_post on b_post.patientunitstayid = s.patientunitstayid",
         " left join vitap_pre c_pre on c_pre.patientunitstayid = s.patientunitstayid",
         " left join vitap_post c_post on c_post.patientunitstayid = s.patientunitstayid;"]

sql = ["\\pset border 0", "\\pset format unaligned",
       "drop table if exists coh2;", "create temp table coh2 (patientunitstayid bigint, t0 numeric);",
       "\\copy coh2 from '%s' with (format csv, header true)" % coh.replace("\\", "/"),
       lab_block("pre"), lab_block("post"), vit_block("pre"), vit_block("post"),
       vitap_block("pre"), vitap_block("post"),
       "\n".join(final),
       "\\copy (select * from n1_eicu_labs) to './n1_eicu_labs.csv' with (format csv, header true)",
       "select count(*) as rows_exported from n1_eicu_labs;"]
p = os.path.join(OUT, "n1_extract_eicu.sql")
open(p, "w", encoding="utf-8").write("\n".join(sql))
print("written:", p)
