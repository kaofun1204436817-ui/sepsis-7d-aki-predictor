\pset border 0
\pset format unaligned
drop table if exists coh;
create temp table coh as
select stay_id, subject_id, hadm_id, t0
from mimiciv_hosp.cn2_s3b_cohort
where cr_base24 is not null;

drop table if exists chem_pre;
create temp table chem_pre as
select s.stay_id,
    min(x.albumin) as alb_pre_min, max(x.albumin) as alb_pre_max,
    min(x.globulin) as glo_pre_min, max(x.globulin) as glo_pre_max,
    min(x.total_protein) as tp_pre_min, max(x.total_protein) as tp_pre_max,
    min(x.aniongap) as agap_pre_min, max(x.aniongap) as agap_pre_max,
    min(x.bicarbonate) as hco3_pre_min, max(x.bicarbonate) as hco3_pre_max,
    min(x.bun) as bun_pre_min, max(x.bun) as bun_pre_max,
    min(x.calcium) as ca_pre_min, max(x.calcium) as ca_pre_max,
    min(x.chloride) as cl_pre_min, max(x.chloride) as cl_pre_max,
    min(x.creatinine) as cr_pre_min, max(x.creatinine) as cr_pre_max,
    min(x.glucose) as glu_pre_min, max(x.glucose) as glu_pre_max,
    min(x.sodium) as na_pre_min, max(x.sodium) as na_pre_max,
    min(x.potassium) as k_pre_min, max(x.potassium) as k_pre_max,
    count(x.albumin) as n_chem_pre
from coh s join mimiciv_derived.chemistry x on x.hadm_id = s.hadm_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists chem_post;
create temp table chem_post as
select s.stay_id,
    min(x.albumin) as alb_post_min, max(x.albumin) as alb_post_max,
    min(x.globulin) as glo_post_min, max(x.globulin) as glo_post_max,
    min(x.total_protein) as tp_post_min, max(x.total_protein) as tp_post_max,
    min(x.aniongap) as agap_post_min, max(x.aniongap) as agap_post_max,
    min(x.bicarbonate) as hco3_post_min, max(x.bicarbonate) as hco3_post_max,
    min(x.bun) as bun_post_min, max(x.bun) as bun_post_max,
    min(x.calcium) as ca_post_min, max(x.calcium) as ca_post_max,
    min(x.chloride) as cl_post_min, max(x.chloride) as cl_post_max,
    min(x.creatinine) as cr_post_min, max(x.creatinine) as cr_post_max,
    min(x.glucose) as glu_post_min, max(x.glucose) as glu_post_max,
    min(x.sodium) as na_post_min, max(x.sodium) as na_post_max,
    min(x.potassium) as k_post_min, max(x.potassium) as k_post_max,
    count(x.albumin) as n_chem_post
from coh s join mimiciv_derived.chemistry x on x.hadm_id = s.hadm_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists cbc_pre;
create temp table cbc_pre as
select s.stay_id,
    min(x.hematocrit) as hct_pre_min, max(x.hematocrit) as hct_pre_max,
    min(x.hemoglobin) as hb_pre_min, max(x.hemoglobin) as hb_pre_max,
    min(x.mch) as mch_pre_min, max(x.mch) as mch_pre_max,
    min(x.mchc) as mchc_pre_min, max(x.mchc) as mchc_pre_max,
    min(x.mcv) as mcv_pre_min, max(x.mcv) as mcv_pre_max,
    min(x.platelet) as plt_pre_min, max(x.platelet) as plt_pre_max,
    min(x.rbc) as rbc_pre_min, max(x.rbc) as rbc_pre_max,
    min(x.rdw) as rdw_pre_min, max(x.rdw) as rdw_pre_max,
    min(x.wbc) as wbc_pre_min, max(x.wbc) as wbc_pre_max,
    count(x.hematocrit) as n_cbc_pre
from coh s join mimiciv_derived.complete_blood_count x on x.hadm_id = s.hadm_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists cbc_post;
create temp table cbc_post as
select s.stay_id,
    min(x.hematocrit) as hct_post_min, max(x.hematocrit) as hct_post_max,
    min(x.hemoglobin) as hb_post_min, max(x.hemoglobin) as hb_post_max,
    min(x.mch) as mch_post_min, max(x.mch) as mch_post_max,
    min(x.mchc) as mchc_post_min, max(x.mchc) as mchc_post_max,
    min(x.mcv) as mcv_post_min, max(x.mcv) as mcv_post_max,
    min(x.platelet) as plt_post_min, max(x.platelet) as plt_post_max,
    min(x.rbc) as rbc_post_min, max(x.rbc) as rbc_post_max,
    min(x.rdw) as rdw_post_min, max(x.rdw) as rdw_post_max,
    min(x.wbc) as wbc_post_min, max(x.wbc) as wbc_post_max,
    count(x.hematocrit) as n_cbc_post
from coh s join mimiciv_derived.complete_blood_count x on x.hadm_id = s.hadm_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists bg_pre;
create temp table bg_pre as
select s.stay_id,
    min(x.lactate) as lac_pre_min, max(x.lactate) as lac_pre_max,
    min(x.ph) as ph_pre_min, max(x.ph) as ph_pre_max,
    min(x.po2) as po2_pre_min, max(x.po2) as po2_pre_max,
    min(x.pco2) as pco2_pre_min, max(x.pco2) as pco2_pre_max,
    min(x.fio2) as fio2_pre_min, max(x.fio2) as fio2_pre_max,
    min(x.pao2fio2ratio) as pf_pre_min, max(x.pao2fio2ratio) as pf_pre_max,
    min(x.baseexcess) as be_pre_min, max(x.baseexcess) as be_pre_max,
    min(x.so2) as so2_pre_min, max(x.so2) as so2_pre_max,
    count(x.lactate) as n_bg_pre
from coh s join mimiciv_derived.bg x on x.hadm_id = s.hadm_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists bg_post;
create temp table bg_post as
select s.stay_id,
    min(x.lactate) as lac_post_min, max(x.lactate) as lac_post_max,
    min(x.ph) as ph_post_min, max(x.ph) as ph_post_max,
    min(x.po2) as po2_post_min, max(x.po2) as po2_post_max,
    min(x.pco2) as pco2_post_min, max(x.pco2) as pco2_post_max,
    min(x.fio2) as fio2_post_min, max(x.fio2) as fio2_post_max,
    min(x.pao2fio2ratio) as pf_post_min, max(x.pao2fio2ratio) as pf_post_max,
    min(x.baseexcess) as be_post_min, max(x.baseexcess) as be_post_max,
    min(x.so2) as so2_post_min, max(x.so2) as so2_post_max,
    count(x.lactate) as n_bg_post
from coh s join mimiciv_derived.bg x on x.hadm_id = s.hadm_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists vit_pre;
create temp table vit_pre as
select s.stay_id,
    min(x.heart_rate) as hr_pre_min, max(x.heart_rate) as hr_pre_max,
    min(x.sbp) as sbp_pre_min, max(x.sbp) as sbp_pre_max,
    min(x.dbp) as dbp_pre_min, max(x.dbp) as dbp_pre_max,
    min(x.mbp) as mbp_pre_min, max(x.mbp) as mbp_pre_max,
    min(x.resp_rate) as rr_pre_min, max(x.resp_rate) as rr_pre_max,
    min(x.temperature) as temp_pre_min, max(x.temperature) as temp_pre_max,
    min(x.spo2) as spo2_pre_min, max(x.spo2) as spo2_pre_max,
    count(x.heart_rate) as n_vit_pre
from coh s join mimiciv_derived.vitalsign x on x.stay_id = s.stay_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists vit_post;
create temp table vit_post as
select s.stay_id,
    min(x.heart_rate) as hr_post_min, max(x.heart_rate) as hr_post_max,
    min(x.sbp) as sbp_post_min, max(x.sbp) as sbp_post_max,
    min(x.dbp) as dbp_post_min, max(x.dbp) as dbp_post_max,
    min(x.mbp) as mbp_post_min, max(x.mbp) as mbp_post_max,
    min(x.resp_rate) as rr_post_min, max(x.resp_rate) as rr_post_max,
    min(x.temperature) as temp_post_min, max(x.temperature) as temp_post_max,
    min(x.spo2) as spo2_post_min, max(x.spo2) as spo2_post_max,
    count(x.heart_rate) as n_vit_post
from coh s join mimiciv_derived.vitalsign x on x.stay_id = s.stay_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists gcs_pre;
create temp table gcs_pre as
select s.stay_id,
    min(x.gcs) as gcs_pre_min, max(x.gcs) as gcs_pre_max,
    count(x.gcs) as n_gcs_pre
from coh s join mimiciv_derived.gcs x on x.stay_id = s.stay_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists gcs_post;
create temp table gcs_post as
select s.stay_id,
    min(x.gcs) as gcs_post_min, max(x.gcs) as gcs_post_max,
    count(x.gcs) as n_gcs_post
from coh s join mimiciv_derived.gcs x on x.stay_id = s.stay_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists uo_pre;
create temp table uo_pre as
select s.stay_id,
    min(x.uo_mlkghr_6hr) as uo6_pre_min, max(x.uo_mlkghr_6hr) as uo6_pre_max,
    min(x.uo_mlkghr_12hr) as uo12_pre_min, max(x.uo_mlkghr_12hr) as uo12_pre_max,
    min(x.uo_mlkghr_24hr) as uo24_pre_min, max(x.uo_mlkghr_24hr) as uo24_pre_max,
    count(x.uo_mlkghr_6hr) as n_uo_pre
from coh s join mimiciv_derived.urine_output_rate x on x.stay_id = s.stay_id and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists uo_post;
create temp table uo_post as
select s.stay_id,
    min(x.uo_mlkghr_6hr) as uo6_post_min, max(x.uo_mlkghr_6hr) as uo6_post_max,
    min(x.uo_mlkghr_12hr) as uo12_post_min, max(x.uo_mlkghr_12hr) as uo12_post_max,
    min(x.uo_mlkghr_24hr) as uo24_post_min, max(x.uo_mlkghr_24hr) as uo24_post_max,
    count(x.uo_mlkghr_6hr) as n_uo_post
from coh s join mimiciv_derived.urine_output_rate x on x.stay_id = s.stay_id and x.charttime > s.t0 and x.charttime <= s.t0 + interval '24 hour'
group by s.stay_id;

drop table if exists crrt_pre;
create table crrt_pre as
select s.stay_id, count(*) as n_crrt_pre
from coh s join mimiciv_derived.crrt x on x.stay_id = s.stay_id
  and x.charttime > s.t0 - interval '24 hour' and x.charttime <= s.t0
group by s.stay_id;

drop table if exists n1_mimic_labs;
create temp table n1_mimic_labs as
select s.stay_id,
  p_chem.alb_pre_min,
  p_chem.alb_pre_max,
  p_chem.glo_pre_min,
  p_chem.glo_pre_max,
  p_chem.tp_pre_min,
  p_chem.tp_pre_max,
  p_chem.agap_pre_min,
  p_chem.agap_pre_max,
  p_chem.hco3_pre_min,
  p_chem.hco3_pre_max,
  p_chem.bun_pre_min,
  p_chem.bun_pre_max,
  p_chem.ca_pre_min,
  p_chem.ca_pre_max,
  p_chem.cl_pre_min,
  p_chem.cl_pre_max,
  p_chem.cr_pre_min,
  p_chem.cr_pre_max,
  p_chem.glu_pre_min,
  p_chem.glu_pre_max,
  p_chem.na_pre_min,
  p_chem.na_pre_max,
  p_chem.k_pre_min,
  p_chem.k_pre_max,
  p_chem.n_chem_pre,
  q_chem.alb_post_min,
  q_chem.alb_post_max,
  q_chem.glo_post_min,
  q_chem.glo_post_max,
  q_chem.tp_post_min,
  q_chem.tp_post_max,
  q_chem.agap_post_min,
  q_chem.agap_post_max,
  q_chem.hco3_post_min,
  q_chem.hco3_post_max,
  q_chem.bun_post_min,
  q_chem.bun_post_max,
  q_chem.ca_post_min,
  q_chem.ca_post_max,
  q_chem.cl_post_min,
  q_chem.cl_post_max,
  q_chem.cr_post_min,
  q_chem.cr_post_max,
  q_chem.glu_post_min,
  q_chem.glu_post_max,
  q_chem.na_post_min,
  q_chem.na_post_max,
  q_chem.k_post_min,
  q_chem.k_post_max,
  q_chem.n_chem_post,
  p_cbc.hct_pre_min,
  p_cbc.hct_pre_max,
  p_cbc.hb_pre_min,
  p_cbc.hb_pre_max,
  p_cbc.mch_pre_min,
  p_cbc.mch_pre_max,
  p_cbc.mchc_pre_min,
  p_cbc.mchc_pre_max,
  p_cbc.mcv_pre_min,
  p_cbc.mcv_pre_max,
  p_cbc.plt_pre_min,
  p_cbc.plt_pre_max,
  p_cbc.rbc_pre_min,
  p_cbc.rbc_pre_max,
  p_cbc.rdw_pre_min,
  p_cbc.rdw_pre_max,
  p_cbc.wbc_pre_min,
  p_cbc.wbc_pre_max,
  p_cbc.n_cbc_pre,
  q_cbc.hct_post_min,
  q_cbc.hct_post_max,
  q_cbc.hb_post_min,
  q_cbc.hb_post_max,
  q_cbc.mch_post_min,
  q_cbc.mch_post_max,
  q_cbc.mchc_post_min,
  q_cbc.mchc_post_max,
  q_cbc.mcv_post_min,
  q_cbc.mcv_post_max,
  q_cbc.plt_post_min,
  q_cbc.plt_post_max,
  q_cbc.rbc_post_min,
  q_cbc.rbc_post_max,
  q_cbc.rdw_post_min,
  q_cbc.rdw_post_max,
  q_cbc.wbc_post_min,
  q_cbc.wbc_post_max,
  q_cbc.n_cbc_post,
  p_bg.lac_pre_min,
  p_bg.lac_pre_max,
  p_bg.ph_pre_min,
  p_bg.ph_pre_max,
  p_bg.po2_pre_min,
  p_bg.po2_pre_max,
  p_bg.pco2_pre_min,
  p_bg.pco2_pre_max,
  p_bg.fio2_pre_min,
  p_bg.fio2_pre_max,
  p_bg.pf_pre_min,
  p_bg.pf_pre_max,
  p_bg.be_pre_min,
  p_bg.be_pre_max,
  p_bg.so2_pre_min,
  p_bg.so2_pre_max,
  p_bg.n_bg_pre,
  q_bg.lac_post_min,
  q_bg.lac_post_max,
  q_bg.ph_post_min,
  q_bg.ph_post_max,
  q_bg.po2_post_min,
  q_bg.po2_post_max,
  q_bg.pco2_post_min,
  q_bg.pco2_post_max,
  q_bg.fio2_post_min,
  q_bg.fio2_post_max,
  q_bg.pf_post_min,
  q_bg.pf_post_max,
  q_bg.be_post_min,
  q_bg.be_post_max,
  q_bg.so2_post_min,
  q_bg.so2_post_max,
  q_bg.n_bg_post,
  p_vit.hr_pre_min,
  p_vit.hr_pre_max,
  p_vit.sbp_pre_min,
  p_vit.sbp_pre_max,
  p_vit.dbp_pre_min,
  p_vit.dbp_pre_max,
  p_vit.mbp_pre_min,
  p_vit.mbp_pre_max,
  p_vit.rr_pre_min,
  p_vit.rr_pre_max,
  p_vit.temp_pre_min,
  p_vit.temp_pre_max,
  p_vit.spo2_pre_min,
  p_vit.spo2_pre_max,
  p_vit.n_vit_pre,
  q_vit.hr_post_min,
  q_vit.hr_post_max,
  q_vit.sbp_post_min,
  q_vit.sbp_post_max,
  q_vit.dbp_post_min,
  q_vit.dbp_post_max,
  q_vit.mbp_post_min,
  q_vit.mbp_post_max,
  q_vit.rr_post_min,
  q_vit.rr_post_max,
  q_vit.temp_post_min,
  q_vit.temp_post_max,
  q_vit.spo2_post_min,
  q_vit.spo2_post_max,
  q_vit.n_vit_post,
  p_gcs.gcs_pre_min,
  p_gcs.gcs_pre_max,
  p_gcs.n_gcs_pre,
  q_gcs.gcs_post_min,
  q_gcs.gcs_post_max,
  q_gcs.n_gcs_post,
  p_uo.uo6_pre_min,
  p_uo.uo6_pre_max,
  p_uo.uo12_pre_min,
  p_uo.uo12_pre_max,
  p_uo.uo24_pre_min,
  p_uo.uo24_pre_max,
  p_uo.n_uo_pre,
  q_uo.uo6_post_min,
  q_uo.uo6_post_max,
  q_uo.uo12_post_min,
  q_uo.uo12_post_max,
  q_uo.uo24_post_min,
  q_uo.uo24_post_max,
  q_uo.n_uo_post,
  cp.n_crrt_pre
from coh s
 left join chem_pre p_chem on p_chem.stay_id = s.stay_id
 left join chem_post q_chem on q_chem.stay_id = s.stay_id
 left join cbc_pre p_cbc on p_cbc.stay_id = s.stay_id
 left join cbc_post q_cbc on q_cbc.stay_id = s.stay_id
 left join bg_pre p_bg on p_bg.stay_id = s.stay_id
 left join bg_post q_bg on q_bg.stay_id = s.stay_id
 left join vit_pre p_vit on p_vit.stay_id = s.stay_id
 left join vit_post q_vit on q_vit.stay_id = s.stay_id
 left join gcs_pre p_gcs on p_gcs.stay_id = s.stay_id
 left join gcs_post q_gcs on q_gcs.stay_id = s.stay_id
 left join uo_pre p_uo on p_uo.stay_id = s.stay_id
 left join uo_post q_uo on q_uo.stay_id = s.stay_id
 left join crrt_pre cp on cp.stay_id = s.stay_id;
\copy (select * from n1_mimic_labs) to './n1_mimic_labs.csv' with (format csv, header true)
select count(*) as rows_exported from n1_mimic_labs;
