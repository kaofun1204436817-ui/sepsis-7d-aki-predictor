# -*- coding: utf-8 -*-
"""gen_n1_sql.py —— 生成 N1 变量提取 SQL（MIMIC-IV），输出 n1_extract_mimic.sql"""
import os
OUT = os.environ.get("CN2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))

SRC = [
    ("chem", "mimiciv_derived.chemistry", "hadm",
     [("albumin", "alb"), ("globulin", "glo"), ("total_protein", "tp"), ("aniongap", "agap"),
      ("bicarbonate", "hco3"), ("bun", "bun"), ("calcium", "ca"), ("chloride", "cl"),
      ("creatinine", "cr"), ("glucose", "glu"), ("sodium", "na"), ("potassium", "k")]),
    ("cbc", "mimiciv_derived.complete_blood_count", "hadm",
     [("hematocrit", "hct"), ("hemoglobin", "hb"), ("mch", "mch"), ("mchc", "mchc"),
      ("mcv", "mcv"), ("platelet", "plt"), ("rbc", "rbc"), ("rdw", "rdw"), ("wbc", "wbc")]),
    ("bg", "mimiciv_derived.bg", "hadm",
     [("lactate", "lac"), ("ph", "ph"), ("po2", "po2"), ("pco2", "pco2"), ("fio2", "fio2"),
      ("pao2fio2ratio", "pf"), ("baseexcess", "be"), ("so2", "so2")]),
    ("vit", "mimiciv_derived.vitalsign", "stay",
     [("heart_rate", "hr"), ("sbp", "sbp"), ("dbp", "dbp"), ("mbp", "mbp"),
      ("resp_rate", "rr"), ("temperature", "temp"), ("spo2", "spo2")]),
    ("gcs", "mimiciv_derived.gcs", "stay", [("gcs", "gcs")]),
    ("uo", "mimiciv_derived.urine_output_rate", "stay",
     [("uo_mlkghr_6hr", "uo6"), ("uo_mlkghr_12hr", "uo12"), ("uo_mlkghr_24hr", "uo24")]),
]

def agg_block(alias, table, key, cols, tag):
    keycol = "x.hadm_id = s.hadm_id" if key == "hadm" else "x.stay_id = s.stay_id"
    if tag == "pre":
        cond = "x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0"
    else:
        cond = "x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'"
    lines = ["    min(x.%s) as %s_%s_min, max(x.%s) as %s_%s_max," % (c, n, tag, c, n, tag)
             for c, n in cols]
    lines.append("    count(x.%s) as n_%s_%s" % (cols[0][0], alias, tag))
    return ("drop table if exists %s_%s;\ncreate temp table %s_%s as\nselect s.stay_id,\n%s\n"
            "from coh s join %s x on %s and %s\ngroup by s.stay_id;\n"
            % (alias, tag, alias, tag, "\n".join(lines), table, keycol, cond))

parts = ["""\\pset border 0
\\pset format unaligned
drop table if exists coh;
create temp table coh as
select stay_id, subject_id, hadm_id, t0
from mimiciv_hosp.cn2_s3b_cohort
where cr_base24 is not null;
"""]
for alias, table, key, cols in SRC:
    for tag in ("pre", "post"):
        parts.append(agg_block(alias, table, key, cols, tag))

parts.append("""drop table if exists crrt_pre;
create table crrt_pre as
select s.stay_id, count(*) as n_crrt_pre
from coh s join mimiciv_derived.crrt x on x.stay_id = s.stay_id
  and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;
""")

final = ["drop table if exists n1_mimic_labs;", "create temp table n1_mimic_labs as", "select s.stay_id,"]
picks = []
for alias, _, _, cols in SRC:
    for tag in ("pre", "post"):
        pref = "p" if tag == "pre" else "q"
        for c, n in cols:
            picks.append("  %s_%s.%s_%s_min" % (pref, alias, n, tag))
            picks.append("  %s_%s.%s_%s_max" % (pref, alias, n, tag))
        picks.append("  %s_%s.n_%s_%s" % (pref, alias, alias, tag))
picks.append("  cp.n_crrt_pre")
final.append(",\n".join(picks))
final.append("from coh s")
for alias, _, _, _ in SRC:
    final.append(" left join %s_pre p_%s on p_%s.stay_id = s.stay_id" % (alias, alias, alias))
    final.append(" left join %s_post q_%s on q_%s.stay_id = s.stay_id" % (alias, alias, alias))
final.append(" left join crrt_pre cp on cp.stay_id = s.stay_id;")
parts.append("\n".join(final))

parts.append(r"""\copy (select * from n1_mimic_labs) to './n1_mimic_labs.csv' with (format csv, header true)
select count(*) as rows_exported from n1_mimic_labs;
""")

sql = "\n".join(parts)
p = os.path.join(OUT, "n1_extract_mimic.sql")
open(p, "w", encoding="utf-8").write(sql)
print("written:", p, len(sql), "chars")
