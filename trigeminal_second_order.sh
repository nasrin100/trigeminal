#!/bin/bash
set -euo pipefail

# TRIGEMINAL SYSTEM TRACTOGRAPHY - Samir Akeb (2022-2023)
# TRIGEMINAL SYSTEM TRACTOGRAPHY - Arnaud Bore (2023-2024)
# TRIGEMINAL SYSTEM TRACTOGRAPHY - Nasrin Rafiei (2025-2026)
#
# SECOND-ORDER ENSEMBLE VERSION
# SUBJECT-BY-SUBJECT VERSION
# COMBO-WISE VERSION:
#   - each step/theta combo is processed separately through filtering/cut/final_length
#   - ONLY final length-filtered bundles are concatenated at the end
#   - this avoids merging very large raw tractograms before filtering
#
# Organized pipeline:
#   0) prepare subject-specific first-order density maps
#   1) prepare second-order seed masks and thalamus ROIs
#   2) run second-order tracking for each (step, theta) combo in ORIG
#   3) prepare second-order MNI ROIs and cut masks
#   4) process each tracking combo separately in MNI space
#   5) concatenate only final length-filtered bundles across combos
#   6) bring final concatenated MNI bundles back to ORIG

usage() {
    cat <<USAGE 1>&2
Usage:
  $(basename "$0") -s <subjects_parent_or_single_subject_dir> -m <ROIs_clean_dir> -o <out_dir>
                   [-f fa_threshold] [-t threads] [-g true|false]
                   [-p step_size] [-e theta_deg]
                   [--npv_spinal_long N] [--npv_spinal_short N] [--npv_thalamus N]

Example:
  bash $(basename "$0") \
    -s /home/local/USHERBROOKE/rafn2101/data/data_test_retest/SUBJECTS_PARENT \
    -m /path/to/ROIs_clean_dir \
    -o /home/local/USHERBROOKE/rafn2101/data/data_test_retest/final_box_spinal \
    -f 0.15 -t 8 -g false \
    --npv_spinal_long 3000 --npv_spinal_short 300 --npv_thalamus 1500
USAGE
    exit 1
}

# -------------------------
# Parse short options first
# -------------------------
s=""
m=""
o=""
f=""
t=""
g=""
p=""
e=""

while getopts ":s:m:o:f:t:g:p:e:" args; do
    case "${args}" in
        s) s=${OPTARG} ;;
        m) m=${OPTARG} ;;
        o) o=${OPTARG} ;;
        f) f=${OPTARG} ;;
        t) t=${OPTARG} ;;
        g) g=${OPTARG} ;;
        p) p=${OPTARG} ;;
        e) e=${OPTARG} ;;
        *) usage ;;
    esac
done
shift $((OPTIND-1))

# -------------------------
# Parse optional long args
# -------------------------
npv_spinal_long_total=3000
npv_spinal_short_total=300
npv_thalamus_total=1500

while [[ $# -gt 0 ]]; do
    case "$1" in
        --npv_spinal_long)
            npv_spinal_long_total="$2"
            shift 2
            ;;
        --npv_spinal_short)
            npv_spinal_short_total="$2"
            shift 2
            ;;
        --npv_thalamus)
            npv_thalamus_total="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" 1>&2
            usage
            ;;
    esac
done

if [[ -z "${s}" || -z "${m}" || -z "${o}" ]]; then
    usage
fi

subject_dir="${s}"
mni_dir="${m}"
out_dir="${o}"
fa_threshold="${f:-0.15}"
nb_threads="${t:-1}"

gpu=""
if [[ -n "${g}" && "${g}" == "true" ]]; then
    gpu="--use_gpu"
fi

# -------------------------
# If you use --npv_thalamus 1500, files are named from_thalamus_npv1500.
# -------------------------
spinal_long_role="from_spinal_track_npv${npv_spinal_long_total}"
spinal_short_role="from_spinal_track_npv${npv_spinal_short_total}"
thalamus_role="from_thalamus_npv${npv_thalamus_total}"

trk_is_empty() {
    local f="$1"

    if [[ ! -f "${f}" ]]; then
        return 0
    fi

    local n_str
    n_str=$(scil_tractogram_count_streamlines "${f}" 2>/dev/null | grep -Eo '[0-9]+' | tail -n 1 || true)

    if [[ -z "${n_str}" || "${n_str}" -eq 0 ]]; then
        return 0
    else
        return 1
    fi
}

mask_has_nonzero_voxels() {
    local f="$1"

    if [[ ! -f "${f}" ]]; then
        return 1
    fi

    local nz
    nz=$(python - <<PY
import nibabel as nib
img = nib.load(r"""${f}""")
print(int((img.get_fdata() > 0).sum()))
PY
)

    [[ "${nz}" -gt 0 ]]
}

get_label_ids_for_cut() {
    local labels_file="$1"

    python - <<PY
import nibabel as nib
import numpy as np
f = r"""${labels_file}"""
vals = np.unique(nib.load(f).get_fdata())
vals = [int(v) for v in vals if v != 0]
if len(vals) < 2:
    print("")
elif len(vals) == 2:
    print(f"{vals[0]} {vals[1]}")
else:
    print(f"{vals[0]} {vals[-1]}")
PY
}

safe_compute_density_map() {
    local in_trk="$1"
    local out_mask="$2"

    if trk_is_empty "${in_trk}"; then
        echo "WARN: ${in_trk} is missing or empty. Skipping density map."
        rm -f "${out_mask}"
        return 0
    fi

    scil_tractogram_compute_density_map \
        "${in_trk}" \
        "${out_mask}" \
        --binary -f

    if ! mask_has_nonzero_voxels "${out_mask}"; then
        echo "WARN: ${out_mask} has no non-zero voxels after density computation. Removing."
        rm -f "${out_mask}"
    fi
}

safe_apply_mask_transform_to_orig() {
    local in_mask="$1"
    local ref_img="$2"
    local warp="$3"
    local affine="$4"
    local out_mask="$5"

    if ! mask_has_nonzero_voxels "${in_mask}"; then
        echo "WARN: ${in_mask} is missing or empty. Skipping mask transform."
        rm -f "${out_mask}"
        return 0
    fi

    antsApplyTransforms \
        -d 3 \
        -i "${in_mask}" \
        -r "${ref_img}" \
        -t "${warp}" \
        -t "${affine}" \
        -o "${out_mask}"

    if ! mask_has_nonzero_voxels "${out_mask}"; then
        echo "WARN: ${out_mask} has no non-zero voxels after transform. Removing."
        rm -f "${out_mask}"
    fi
}

safe_build_cut_labels() {
    local mask_a="$1"
    local mask_b="$2"
    local out_mask="$3"
    local out_labels="$4"

    if ! mask_has_nonzero_voxels "${mask_a}"; then
        echo "WARN: ${mask_a} is missing or empty. Skipping cut-mask construction."
        rm -f "${out_mask}" "${out_labels}"
        return 0
    fi

    if ! mask_has_nonzero_voxels "${mask_b}"; then
        echo "WARN: ${mask_b} is missing or empty. Skipping cut-mask construction."
        rm -f "${out_mask}" "${out_labels}"
        return 0
    fi

    scil_volume_math union \
        "${mask_a}" \
        "${mask_b}" \
        "${out_mask}" \
        --data_type uint8 -f

    if ! mask_has_nonzero_voxels "${out_mask}"; then
        echo "WARN: ${out_mask} has no non-zero voxels after union. Removing."
        rm -f "${out_mask}" "${out_labels}"
        return 0
    fi

    scil_labels_from_mask \
        "${out_mask}" \
        "${out_labels}" \
        -f
}

safe_filter_by_roi() {
    local in_trk="$1"
    local out_trk="$2"
    shift 2

    if trk_is_empty "${in_trk}"; then
        echo "WARN: ${in_trk} is missing or empty. Skipping filter."
        rm -f "${out_trk}"
        return 0
    fi

    local args=("$@")
    local i=0
    local n=${#args[@]}

    while (( i < n )); do
        local key="${args[$i]}"

        if [[ "${key}" == "--drawn_roi" ]]; then
            if (( i + 3 >= n )); then
                echo "WARN: malformed --drawn_roi arguments for ${out_trk}. Skipping filter."
                rm -f "${out_trk}"
                return 0
            fi

            local roi_file="${args[$((i + 1))]}"
            local roi_action="${args[$((i + 3))]}"

            if [[ ! -f "${roi_file}" ]]; then
                echo "WARN: ROI ${roi_file} is missing. Skipping filter for ${out_trk}."
                rm -f "${out_trk}"
                return 0
            fi

            if [[ "${roi_action}" == "include" ]] && ! mask_has_nonzero_voxels "${roi_file}"; then
                echo "WARN: include ROI ${roi_file} is empty. Skipping filter for ${out_trk}."
                rm -f "${out_trk}"
                return 0
            fi

            i=$((i + 4))
            continue
        fi

        if [[ "${key}" == "--bdo" ]]; then
            if (( i + 3 >= n )); then
                echo "WARN: malformed --bdo arguments for ${out_trk}. Skipping filter."
                rm -f "${out_trk}"
                return 0
            fi

            local bdo_file="${args[$((i + 1))]}"

            if [[ ! -f "${bdo_file}" ]]; then
                echo "WARN: BDO ${bdo_file} is missing. Skipping filter for ${out_trk}."
                rm -f "${out_trk}"
                return 0
            fi

            i=$((i + 4))
            continue
        fi

        i=$((i + 1))
    done

    scil_tractogram_filter_by_roi \
        "${in_trk}" \
        "${out_trk}" \
        "${args[@]}" \
        -f

    if trk_is_empty "${out_trk}"; then
        echo "WARN: ${out_trk} is empty after filtering. Removing."
        rm -f "${out_trk}"
    fi
}

safe_filter_by_length() {
    local in_trk="$1"
    local out_trk="$2"
    local minL="$3"
    local maxL="$4"

    if trk_is_empty "${in_trk}"; then
        echo "WARN: ${in_trk} is missing or empty. Skipping length filtering."
        rm -f "${out_trk}"
        return 0
    fi

    echo "Length filtering ${in_trk}"
    echo "  minL=${minL} maxL=${maxL}"
    echo "  output=${out_trk}"

    scil_tractogram_filter_by_length \
        "${in_trk}" \
        "${out_trk}" \
        --minL "${minL}" \
        --maxL "${maxL}" \
        --display_counts -f

    if trk_is_empty "${out_trk}"; then
        echo "WARN: ${out_trk} is empty after length filtering. Removing."
        rm -f "${out_trk}"
    fi
}

safe_concatenate_incremental() {
    local out_trk="$1"
    shift

    local files=("$@")

    if (( ${#files[@]} == 0 )); then
        echo "WARN: no files given to concatenate for ${out_trk}"
        rm -f "${out_trk}"
        return 0
    fi

    if (( ${#files[@]} == 1 )); then
        cp -f "${files[0]}" "${out_trk}"
        return 0
    fi

    local tmp_dir
    tmp_dir="$(dirname "${out_trk}")/tmp_concat_$(basename "${out_trk}" .trk)"
    mkdir -p "${tmp_dir}"

    local tmp_merge="${tmp_dir}/merge_0.trk"
    cp -f "${files[0]}" "${tmp_merge}"

    local idx=1
    local cf
    for cf in "${files[@]:1}"; do
        local next_tmp="${tmp_dir}/merge_${idx}.trk"

        echo "  Concatenate ${idx}/${#files[@]} into $(basename "${out_trk}")"

        scil_tractogram_math concatenate \
            "${tmp_merge}" \
            "${cf}" \
            "${next_tmp}" \
            -f

        rm -f "${tmp_merge}"
        tmp_merge="${next_tmp}"
        idx=$((idx + 1))
    done

    mv -f "${tmp_merge}" "${out_trk}"
    rm -rf "${tmp_dir}"

    if trk_is_empty "${out_trk}"; then
        echo "WARN: ${out_trk} is empty after final concatenation. Removing."
        rm -f "${out_trk}"
    fi
}

# -------------------------
# Ensemble grid
# -------------------------
if [[ -n "${p}" && -n "${e}" ]]; then
    step_list=("${p}")
    theta_list=("${e}")
else
    step_list=(0.1 0.5 1.0)
    theta_list=(20 30 40)
fi

n_combos=$(( ${#step_list[@]} * ${#theta_list[@]} ))

npv_spinal_long_per_combo=$(( (npv_spinal_long_total + n_combos - 1) / n_combos ))
npv_spinal_short_per_combo=$(( (npv_spinal_short_total + n_combos - 1) / n_combos ))
npv_thalamus_per_combo=$(( (npv_thalamus_total + n_combos - 1) / n_combos ))

echo "Folder subjects: ${subject_dir}"
echo "Folder MNI: ${mni_dir}"
echo "Output folder: ${out_dir}"
echo "FA threshold: ${fa_threshold}"
echo "GPU: ${gpu}"
echo "Threads: ${nb_threads}"
echo "Tracking grid: steps=${step_list[*]}  thetas=${theta_list[*]}"
echo "Second-order budgets per combo:"
echo "  spinal long:  ${npv_spinal_long_per_combo}  (from total ${npv_spinal_long_total})"
echo "  spinal short: ${npv_spinal_short_per_combo} (from total ${npv_spinal_short_total})"
echo "  thalamus:     ${npv_thalamus_per_combo}     (from total ${npv_thalamus_total})"
echo "Role names:"
echo "  spinal long role:  ${spinal_long_role}"
echo "  spinal short role: ${spinal_short_role}"
echo "  thalamus role:     ${thalamus_role}"

export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="${nb_threads}"

if [[ -d "${subject_dir}/tractoflow" ]]; then
    subject_list=("${subject_dir}")
else
    shopt -s nullglob
    subject_list=("${subject_dir}"/*/)
    shopt -u nullglob
fi

if [[ ${#subject_list[@]} -eq 0 ]]; then
    echo "ERROR: No subject folders found in ${subject_dir}"
    exit 1
fi

for nsub_path in "${subject_list[@]}"; do
    nsub=$(basename "${nsub_path%/}")

    mkdir -p "${out_dir}/${nsub}/orig_space/rois"
    mkdir -p "${out_dir}/${nsub}/orig_space/tracking_second_order"/{trials,final}
    mkdir -p "${out_dir}/${nsub}/mni_space/rois"
    mkdir -p "${out_dir}/${nsub}/mni_space/tracking_second_order"/{combo_outputs,final_ensemble}

    orig_rois_dir="${out_dir}/${nsub}/orig_space/rois"
    mni_rois_dir="${out_dir}/${nsub}/mni_space/rois"
    orig_tracking_dir="${out_dir}/${nsub}/orig_space/tracking_second_order"
    mni_tracking_dir_second_order="${out_dir}/${nsub}/mni_space/tracking_second_order"
    first_order_final_dir="${out_dir}/${nsub}/mni_space/tracking_first_order/final_merged/final"

    orig_trials_root="${orig_tracking_dir}/trials"

    echo ""
    echo "|------------- PROCESSING SECOND-ORDER COMBO-WISE ENSEMBLE FOR ${nsub} -------------|"
    echo ""

    for nside in left right; do
        if trk_is_empty "${first_order_final_dir}/${nsub}_${nside}_spinal.trk"; then
            echo "WARN: Missing or empty first-order spinal file: ${first_order_final_dir}/${nsub}_${nside}_spinal.trk"
            echo "WARN: ${nside} spinal-based second-order tracking may be skipped."
        fi

        if trk_is_empty "${first_order_final_dir}/${nsub}_${nside}_remaining_cp.trk"; then
            echo "WARN: Missing or empty first-order remaining_cp file: ${first_order_final_dir}/${nsub}_${nside}_remaining_cp.trk"
            echo "WARN: ${nside} remaining_cp-dependent second-order bundles may be skipped."
        fi
    done

    [[ -f "${orig_rois_dir}/${nsub}_aparc.DKTatlas+aseg_orig.nii.gz" ]] || {
        echo "ERROR: Missing ${orig_rois_dir}/${nsub}_aparc.DKTatlas+aseg_orig.nii.gz"
        exit 1
    }

    [[ -f "${mni_rois_dir}/${nsub}_aparc.DKTatlas+aseg_mni.nii.gz" ]] || {
        echo "ERROR: Missing ${mni_rois_dir}/${nsub}_aparc.DKTatlas+aseg_mni.nii.gz"
        exit 1
    }

    [[ -f "${orig_rois_dir}/${nsub}_wm_mask_${fa_threshold}_orig.nii.gz" ]] || {
        echo "ERROR: Missing ${orig_rois_dir}/${nsub}_wm_mask_${fa_threshold}_orig.nii.gz"
        echo "Check that -f matches the first-order FA threshold."
        exit 1
    }

    [[ -f "${out_dir}/${nsub}/orig_space/transfo/2orig_0GenericAffine.mat" ]] || {
        echo "ERROR: Missing affine transform"
        exit 1
    }

    [[ -f "${out_dir}/${nsub}/orig_space/transfo/2orig_1Warp.nii.gz" ]] || {
        echo "ERROR: Missing forward warp"
        exit 1
    }

    [[ -f "${out_dir}/${nsub}/orig_space/transfo/2orig_1InverseWarp.nii.gz" ]] || {
        echo "ERROR: Missing inverse warp"
        exit 1
    }

    # -------------------------
    # 0) Prepare subject-specific first-order density maps
    # -------------------------
    echo "|------------- 0) Prepare subject-specific first-order density maps -------------|"
    for nside in left right; do
        safe_compute_density_map \
            "${first_order_final_dir}/${nsub}_${nside}_spinal.trk" \
            "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz"

        safe_compute_density_map \
            "${first_order_final_dir}/${nsub}_${nside}_remaining_cp.trk" \
            "${mni_rois_dir}/${nsub}_${nside}_remaining_cp_density_mni.nii.gz"
    done

    # -------------------------
    # 1) Prepare second-order seed masks and thalamus ROIs
    # -------------------------
    echo "|------------- 1) Prepare second-order seed masks and thalamus ROIs -------------|"

    echo "|------------- 1.1) Transform subject spinal seed masks to orig space -------------|"
    for nside in left right; do
        safe_apply_mask_transform_to_orig \
            "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" \
            "${nsub_path}/tractoflow/${nsub}__t1_warped.nii.gz" \
            "${out_dir}/${nsub}/orig_space/transfo/2orig_1Warp.nii.gz" \
            "${out_dir}/${nsub}/orig_space/transfo/2orig_0GenericAffine.mat" \
            "${orig_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_orig.nii.gz"
    done

    echo "|------------- 1.2) Extract thalamus masks from the anatomical segmentation -------------|"
    Right_Thalamus=(49)
    Left_Thalamus=(10)

    scil_labels_combine "${orig_rois_dir}/${nsub}_right_thalamus_orig.nii.gz" \
        --volume_ids "${orig_rois_dir}/${nsub}_aparc.DKTatlas+aseg_orig.nii.gz" ${Right_Thalamus[*]} \
        --merge_groups -f

    scil_labels_combine "${mni_rois_dir}/${nsub}_right_thalamus_mni.nii.gz" \
        --volume_ids "${mni_rois_dir}/${nsub}_aparc.DKTatlas+aseg_mni.nii.gz" ${Right_Thalamus[*]} \
        --merge_groups -f

    scil_labels_combine "${orig_rois_dir}/${nsub}_left_thalamus_orig.nii.gz" \
        --volume_ids "${orig_rois_dir}/${nsub}_aparc.DKTatlas+aseg_orig.nii.gz" ${Left_Thalamus[*]} \
        --merge_groups -f

    scil_labels_combine "${mni_rois_dir}/${nsub}_left_thalamus_mni.nii.gz" \
        --volume_ids "${mni_rois_dir}/${nsub}_aparc.DKTatlas+aseg_mni.nii.gz" ${Left_Thalamus[*]} \
        --merge_groups -f

    # -------------------------
    # 2) Run second-order ensemble tracking in orig space
    # -------------------------
    echo "|------------- 2) Run second-order ensemble tracking in orig space -------------|"
    for step_size in "${step_list[@]}"; do
        for theta in "${theta_list[@]}"; do
            combo_tag="step_${step_size}_theta_${theta}"
            mkdir -p "${orig_trials_root}/${combo_tag}"

            echo "|=== Second-order combo: ${combo_tag} ===|"

            for nside in left right; do
                spinal_seed="${orig_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_orig.nii.gz"
                thalamus_seed="${orig_rois_dir}/${nsub}_${nside}_thalamus_orig.nii.gz"
                wm_mask="${orig_rois_dir}/${nsub}_wm_mask_${fa_threshold}_orig.nii.gz"

                if mask_has_nonzero_voxels "${spinal_seed}"; then
                    scil_tracking_local \
                        "${nsub_path}/tractoflow/${nsub}__fodf.nii.gz" \
                        "${spinal_seed}" \
                        "${wm_mask}" \
                        "${orig_trials_root}/${combo_tag}/${nsub}_${nside}_${spinal_long_role}_${combo_tag}.trk" \
                        --npv "${npv_spinal_long_per_combo}" \
                        --step "${step_size}" \
                        --theta "${theta}" \
                        ${gpu} -v -f

                    scil_tracking_local \
                        "${nsub_path}/tractoflow/${nsub}__fodf.nii.gz" \
                        "${spinal_seed}" \
                        "${wm_mask}" \
                        "${orig_trials_root}/${combo_tag}/${nsub}_${nside}_${spinal_short_role}_${combo_tag}.trk" \
                        --npv "${npv_spinal_short_per_combo}" \
                        --step "${step_size}" \
                        --theta "${theta}" \
                        ${gpu} -v -f
                else
                    echo "WARN: spinal seed is missing or empty for ${nside} at ${combo_tag}. Skipping spinal tracking."
                fi

                if mask_has_nonzero_voxels "${thalamus_seed}"; then
                    scil_tracking_local \
                        "${nsub_path}/tractoflow/${nsub}__fodf.nii.gz" \
                        "${thalamus_seed}" \
                        "${wm_mask}" \
                        "${orig_trials_root}/${combo_tag}/${nsub}_${nside}_${thalamus_role}_${combo_tag}.trk" \
                        --npv "${npv_thalamus_per_combo}" \
                        --step "${step_size}" \
                        --theta "${theta}" \
                        ${gpu} -v -f
                else
                    echo "WARN: thalamus seed is missing or empty for ${nside} at ${combo_tag}. Skipping thalamus tracking."
                fi
            done
        done
    done

    # -------------------------
    # 3) Prepare second-order MNI ROIs and cut masks
    # -------------------------
    echo "|------------- 3) Prepare second-order MNI ROIs and cut masks -------------|"
    echo "|------------- 3.1) Copy VPM and pathway-specific MNI ROIs -------------|"

    for nside in left right; do
        cp "${mni_dir}/MNI/Distal/${nside}/VPM.nii.gz" \
           "${mni_rois_dir}/${nsub}_${nside}_VPM_mni.nii.gz"

        for nroi in "${mni_dir}/MNI/from_${nside}"/*.nii.gz; do
            [[ -f "${nroi}" ]] || continue
            ROI_basename=$(basename "${nroi}")
            cp "${nroi}" "${mni_rois_dir}/${nsub}_second_order_${ROI_basename/nii/_mni.nii}"
        done
    done

    echo "|------------- 3.2) Build cut masks for second-order pathway extraction -------------|"
    for nside in left right; do
        if [[ "${nside}" == "left" ]]; then
            contra_nside="right"
        else
            contra_nside="left"
        fi

        safe_build_cut_labels \
            "${mni_rois_dir}/${nsub}_${nside}_VPM_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_remaining_cp_density_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Ipsilat_dPSN_Cuts_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Ipsilat_dPSN_Cuts_labels_mni.nii.gz"

        safe_build_cut_labels \
            "${mni_rois_dir}/${nsub}_${nside}_VPM_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Ipsilat_CS_Cuts_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Ipsilat_CS_Cuts_labels_mni.nii.gz"

        safe_build_cut_labels \
            "${mni_rois_dir}/${nsub}_${contra_nside}_thalamus_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Controlat_CS_Cuts_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_DTTT_Controlat_CS_Cuts_labels_mni.nii.gz"

        safe_build_cut_labels \
            "${mni_rois_dir}/${nsub}_${contra_nside}_thalamus_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_VTTT_Controlat_OSandIS_Cuts_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_VTTT_Controlat_OSandIS_Cuts_labels_mni.nii.gz"

        safe_build_cut_labels \
            "${mni_rois_dir}/${nsub}_${contra_nside}_VPM_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_remaining_cp_density_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_VTTT_Controlat_vPSN_Cuts_mni.nii.gz" \
            "${mni_rois_dir}/${nsub}_${nside}_second_order_VTTT_Controlat_vPSN_Cuts_labels_mni.nii.gz"
    done

    # -------------------------
    # 4) Process each combo separately
    # -------------------------
    echo "|------------- 4) Process each combo separately before final concatenation -------------|"

    declare -A minL_by_bundle
    declare -A maxL_by_bundle

    minL_by_bundle["DTTT_Ipsilat_CS"]=0
    maxL_by_bundle["DTTT_Ipsilat_CS"]=100

    minL_by_bundle["DTTT_Ipsilat_dPSN"]=0
    maxL_by_bundle["DTTT_Ipsilat_dPSN"]=90

    minL_by_bundle["DTTT_Controlat_CS"]=0
    maxL_by_bundle["DTTT_Controlat_CS"]=165

    minL_by_bundle["VTTT_Controlat_OSandIS"]=0
    maxL_by_bundle["VTTT_Controlat_OSandIS"]=185

    minL_by_bundle["VTTT_Controlat_vPSN"]=0
    maxL_by_bundle["VTTT_Controlat_vPSN"]=90

    for step_size in "${step_list[@]}"; do
        for theta in "${theta_list[@]}"; do
            combo_tag="step_${step_size}_theta_${theta}"
            combo_mni_dir="${mni_tracking_dir_second_order}/combo_outputs/${combo_tag}"

            mkdir -p "${combo_mni_dir}"/{orig,filtered,cut,final,final_length}

            echo ""
            echo "|------------- Processing combo ${combo_tag} for ${nsub} -------------|"
            echo ""

            # -------------------------
            # 4.1) Register this combo's raw tractograms to MNI
            # -------------------------
            echo "|------------- 4.1) Register combo ${combo_tag} tractograms to MNI -------------|"
            for nside in left right; do
                for role in "${thalamus_role}" "${spinal_short_role}" "${spinal_long_role}"; do
                    in_trk="${orig_trials_root}/${combo_tag}/${nsub}_${nside}_${role}_${combo_tag}.trk"
                    out_trk="${combo_mni_dir}/orig/${nsub}_${nside}_${role}_${combo_tag}.trk"

                    if trk_is_empty "${in_trk}"; then
                        echo "WARN: raw combo tractogram missing or empty: ${in_trk}"
                        rm -f "${out_trk}"
                        continue
                    fi

                    scil_tractogram_apply_transform \
                        "${in_trk}" \
                        "${mni_dir}/MNI/mni_masked.nii.gz" \
                        "${out_dir}/${nsub}/orig_space/transfo/2orig_0GenericAffine.mat" \
                        "${out_trk}" \
                        --in_deformation "${out_dir}/${nsub}/orig_space/transfo/2orig_1Warp.nii.gz" \
                        --remove_invalid \
                        --reverse_operation -f

                    if trk_is_empty "${out_trk}"; then
                        echo "WARN: ${out_trk} is empty after transform to MNI. Removing."
                        rm -f "${out_trk}"
                    fi
                done
            done

            # -------------------------
            # 4.2) Filter this combo's tractograms into pathway bundles
            # -------------------------
            echo "|------------- 4.2) Filter combo ${combo_tag} into pathway bundles -------------|"
            for nside in left right; do
                if [[ "${nside}" == "left" ]]; then
                    contra_nside="right"
                else
                    contra_nside="left"
                fi

                echo "|------------- 4.2.1) Combo ${combo_tag}: from ${nside} - VTTT contralateral OS/IS -------------|"
                safe_filter_by_roi \
                    "${combo_mni_dir}/orig/${nsub}_${contra_nside}_${thalamus_role}_${combo_tag}.trk" \
                    "${combo_mni_dir}/filtered/${nsub}_from_${nside}_VTTT_Controlat_OSandIS_${combo_tag}.trk" \
                    --drawn_roi "${mni_rois_dir}/${nsub}_left_cerebellum_wm_mni.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_right_cerebellum_wm_mni.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${contra_nside}_thalamus_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_INC_Pons_Controlat.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_Ventral_Brainstem.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_CaudalMedulla_Controlat.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_INC_VTT_Area.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_Pons_Ipsilat.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/cs_plaque.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_INC_VTT_Area.nii.gz" 'any' 'include' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/VTTT_Controlat_OSandIS_1.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/VTTT_Controlat_OSandIS_2.bdo" 'any' 'exclude'

                echo "|------------- 4.2.2) Combo ${combo_tag}: from ${nside} - VTTT contralateral vPSN -------------|"
                safe_filter_by_roi \
                    "${combo_mni_dir}/orig/${nsub}_${nside}_${spinal_long_role}_${combo_tag}.trk" \
                    "${combo_mni_dir}/filtered/${nsub}_from_${nside}_VTTT_Controlat_vPSN_${combo_tag}.trk" \
                    --drawn_roi "${mni_rois_dir}/${nsub}_left_cerebellum_wm_mni.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_right_cerebellum_wm_mni.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" 'either_end' 'include' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${contra_nside}_VPM_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_Ventral_Brainstem.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_CaudalMedulla_Controlat.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_INC_VTT_Area.nii.gz" 'any' 'include' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_1.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_2.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_3.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_4.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_5.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/VTTT_Controlat/VTTT_Controlat_vPSN_6.bdo" 'any' 'exclude'

                echo "|------------- 4.2.3) Combo ${combo_tag}: ${nside} - DTTT contralateral CS -------------|"
                safe_filter_by_roi \
                    "${combo_mni_dir}/orig/${nsub}_${nside}_${thalamus_role}_${combo_tag}.trk" \
                    "${combo_mni_dir}/filtered/${nsub}_from_${contra_nside}_DTTT_Controlat_CS_${combo_tag}.trk" \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${contra_nside}_spinal_density_second_order_seed_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_thalamus_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${contra_nside}/VTTT_Controlat_EXC_Ventral_Brainstem.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/from_${contra_nside}/DTTT_Controlat_INC_CaudalMedulla_Ipsilat.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${contra_nside}/DTTT_Controlat_INC_Medulla_Controlat.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${contra_nside}/DTTT_Controlat_EXC_Midbrain_Ipsilat.nii.gz" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Controlat_1.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Controlat_2.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Controlat_3.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Controlat_4.bdo" 'any' 'exclude'

                echo "|------------- 4.2.4) Combo ${combo_tag}: ${nside} - DTTT ipsilateral CS -------------|"
                safe_filter_by_roi \
                    "${combo_mni_dir}/orig/${nsub}_${nside}_${spinal_short_role}_${combo_tag}.trk" \
                    "${combo_mni_dir}/filtered/${nsub}_from_${nside}_DTTT_Ipsilat_CS_${combo_tag}.trk" \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_spinal_density_second_order_seed_mni.nii.gz" 'either_end' 'include' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_VPM_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_dir}/MNI/from_${nside}/VTTT_Controlat_EXC_Ventral_Brainstem.nii.gz" 'any' 'exclude' \
                    --drawn_roi "${mni_dir}/MNI/midsagittal_plane.nii.gz" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_CS_1.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_CS_2.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_CS_3.bdo" 'any' 'exclude'

                echo "|------------- 4.2.5) Combo ${combo_tag}: ${nside} - DTTT ipsilateral dPSN -------------|"
                safe_filter_by_roi \
                    "${combo_mni_dir}/orig/${nsub}_${nside}_${spinal_long_role}_${combo_tag}.trk" \
                    "${combo_mni_dir}/filtered/${nsub}_from_${nside}_DTTT_Ipsilat_dPSN_${combo_tag}.trk" \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_VPM_mni.nii.gz" 'any' 'include' \
                    --drawn_roi "${mni_rois_dir}/${nsub}_${nside}_remaining_cp_density_mni.nii.gz" 'either_end' 'include' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_1.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_2.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_3.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_4.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_5.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_6.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_7.bdo" 'any' 'exclude' \
                    --bdo "${mni_dir}/MNI/from_${nside}/new_ROIs/DTTT_Ipsilat_dPSN_8.bdo" 'any' 'exclude'
            done

            # -------------------------
            # 4.3) Cut this combo's filtered bundles with label masks
            # -------------------------
            echo "|------------- 4.3) Cut combo ${combo_tag} filtered bundles -------------|"
            for nside in left right; do
                for nbundle in \
                    VTTT_Controlat_OSandIS \
                    VTTT_Controlat_vPSN \
                    DTTT_Controlat_CS \
                    DTTT_Ipsilat_CS \
                    DTTT_Ipsilat_dPSN; do

                    in_trk="${combo_mni_dir}/filtered/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"
                    out_trk="${combo_mni_dir}/cut/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"
                    labels_file="${mni_rois_dir}/${nsub}_${nside}_second_order_${nbundle}_Cuts_labels_mni.nii.gz"

                    if trk_is_empty "${in_trk}"; then
                        echo "WARN: ${in_trk} is missing or empty, skipping cut."
                        continue
                    fi

                    if [[ ! -f "${labels_file}" ]]; then
                        echo "WARN: Missing labels file ${labels_file}, skipping."
                        continue
                    fi

                    label_ids=$(get_label_ids_for_cut "${labels_file}")

                    if [[ -z "${label_ids}" ]]; then
                        echo "WARN: Could not determine 2 valid label ids for ${labels_file}, skipping."
                        continue
                    fi

                    echo "Cutting combo ${combo_tag}: ${nside} ${nbundle} with label ids: ${label_ids}"

                    scil_tractogram_cut_streamlines \
                        "${in_trk}" \
                        --labels "${labels_file}" \
                        --label_ids ${label_ids} \
                        "${out_trk}" -f

                    if trk_is_empty "${out_trk}"; then
                        echo "WARN: ${out_trk} is empty after cut, removing."
                        rm -f "${out_trk}"
                    fi
                done
            done

            # -------------------------
            # 4.4) Reject outliers for this combo
            # -------------------------
            echo "|------------- 4.4) Reject outliers for combo ${combo_tag} -------------|"
            for nside in left right; do
                in_trk="${combo_mni_dir}/cut/${nsub}_from_${nside}_DTTT_Ipsilat_CS_${combo_tag}.trk"
                out_trk="${combo_mni_dir}/final/${nsub}_from_${nside}_DTTT_Ipsilat_CS_${combo_tag}.trk"

                if trk_is_empty "${in_trk}"; then
                    echo "WARN: ${in_trk} missing or empty, skipping outlier rejection."
                else
                    scil_bundle_reject_outliers \
                        "${in_trk}" \
                        "${out_trk}" \
                        --alpha 0.30 -f

                    if trk_is_empty "${out_trk}"; then
                        echo "WARN: ${out_trk} is empty after outlier rejection, removing."
                        rm -f "${out_trk}"
                    fi
                fi

                for nbundle in DTTT_Ipsilat_dPSN DTTT_Controlat_CS VTTT_Controlat_OSandIS VTTT_Controlat_vPSN; do
                    in_trk="${combo_mni_dir}/cut/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"
                    out_trk="${combo_mni_dir}/final/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"

                    if trk_is_empty "${in_trk}"; then
                        echo "WARN: ${in_trk} missing or empty, skipping outlier rejection."
                    else
                        scil_bundle_reject_outliers \
                            "${in_trk}" \
                            "${out_trk}" \
                            --alpha 0.50 -f

                        if trk_is_empty "${out_trk}"; then
                            echo "WARN: ${out_trk} is empty after outlier rejection, removing."
                            rm -f "${out_trk}"
                        fi
                    fi
                done
            done

            # -------------------------
            # 4.5) Filter final combo bundles by length
            # -------------------------
            echo "|------------- 4.5) Length-filter final combo ${combo_tag} bundles -------------|"
            for nside in left right; do
                for nbundle in \
                    DTTT_Ipsilat_CS \
                    DTTT_Ipsilat_dPSN \
                    DTTT_Controlat_CS \
                    VTTT_Controlat_OSandIS \
                    VTTT_Controlat_vPSN; do

                    in_trk="${combo_mni_dir}/final/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"
                    out_trk="${combo_mni_dir}/final_length/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"

                    minL="${minL_by_bundle[${nbundle}]}"
                    maxL="${maxL_by_bundle[${nbundle}]}"

                    safe_filter_by_length \
                        "${in_trk}" \
                        "${out_trk}" \
                        "${minL}" \
                        "${maxL}"
                done
            done
        done
    done

    # -------------------------
    # 5) Final concatenate across combos
    # -------------------------
    echo "|------------- 5) Final concatenate across combos -------------|"

    mkdir -p "${mni_tracking_dir_second_order}/final_ensemble"

    for nside in left right; do
        for nbundle in \
            DTTT_Ipsilat_CS \
            DTTT_Ipsilat_dPSN \
            DTTT_Controlat_CS \
            VTTT_Controlat_OSandIS \
            VTTT_Controlat_vPSN; do

            files=()

            for step_size in "${step_list[@]}"; do
                for theta in "${theta_list[@]}"; do
                    combo_tag="step_${step_size}_theta_${theta}"
                    combo_mni_dir="${mni_tracking_dir_second_order}/combo_outputs/${combo_tag}"
                    f="${combo_mni_dir}/final_length/${nsub}_from_${nside}_${nbundle}_${combo_tag}.trk"

                    if ! trk_is_empty "${f}"; then
                        files+=("${f}")
                    fi
                done
            done

            out_trk="${mni_tracking_dir_second_order}/final_ensemble/${nsub}_from_${nside}_${nbundle}.trk"

            if (( ${#files[@]} == 0 )); then
                echo "WARN: no final length-filtered files for ${nside} ${nbundle}"
                rm -f "${out_trk}"
                continue
            fi

            echo "Final concatenation: ${#files[@]} combo-final files for ${nside} ${nbundle}"
            safe_concatenate_incremental "${out_trk}" "${files[@]}"
            scil_tractogram_count_streamlines "${out_trk}" || true
        done
    done

    # -------------------------
    # 6) [BACK-TO-ORIG] Register final concatenated MNI bundles to orig space
    # -------------------------
    echo "|------------- 6) [BACK-TO-ORIG] Register final concatenated MNI bundles to orig space -------------|"

    mkdir -p "${orig_tracking_dir}/final"

    for nside in left right; do
        for nbundle in DTTT_Ipsilat_CS DTTT_Ipsilat_dPSN DTTT_Controlat_CS VTTT_Controlat_OSandIS VTTT_Controlat_vPSN; do
            in_trk="${mni_tracking_dir_second_order}/final_ensemble/${nsub}_from_${nside}_${nbundle}.trk"
            out_trk="${orig_tracking_dir}/final/${nsub}_from_${nside}_${nbundle}_orig.trk"

            if trk_is_empty "${in_trk}"; then
                echo "WARN: Final concatenated MNI bundle missing or empty for ${nside} ${nbundle}, skipping back-to-orig."
                rm -f "${out_trk}"
                continue
            fi

            scil_tractogram_apply_transform \
                "${in_trk}" \
                "${nsub_path}/tractoflow/${nsub}__t1_warped.nii.gz" \
                "${out_dir}/${nsub}/orig_space/transfo/2orig_0GenericAffine.mat" \
                "${out_trk}" \
                --inverse \
                --in_deformation "${out_dir}/${nsub}/orig_space/transfo/2orig_1InverseWarp.nii.gz" \
                --remove_invalid -f

            if trk_is_empty "${out_trk}"; then
                echo "WARN: ${out_trk} is empty after back-to-orig transform. Removing."
                rm -f "${out_trk}"
            fi
        done
    done

    echo "|------------- SECOND-ORDER COMBO-WISE ENSEMBLE FOR ${nsub} IS COMPLETED -------------|"
    echo ""
done

# This version avoids the old heavy raw-tractogram merge.
# It processes each step/theta combo separately and concatenates only final filtered bundles..