function batch_preprocess_people()
foot_db_root = "E:\毕设\Foot_database";
out_root = "E:\毕设\Terra-main\people_database\interim_matlab";
datasets = ["A1","A2","A3","A4","A5"];
for d = 1:numel(datasets)
    ds = datasets(d);
    subsets = local_subsets(ds);
    for s = 1:numel(subsets)
        subset = subsets(s);
        subset_in = local_subset_in(foot_db_root, ds, subset);
        subset_out = fullfile(out_root, ds, subset);
        if ~exist(subset_out, "dir")
            mkdir(subset_out);
        end
        local_merge_people(subset_in);
        local_extract_events(subset_in, subset_out);
    end
end
end

function subsets = local_subsets(ds)
if ds == "A2"
    subsets = ["A2_1","A2_2","A2_3"];
elseif ds == "A3"
    subsets = ["A3_1","A3_2","A3_3"];
elseif ds == "A5"
    subsets = ["A5_1","A5_2","A5_3"];
else
    subsets = ds;
end
end

function p = local_subset_in(foot_db_root, ds, subset)
if ds == "A2" || ds == "A3" || ds == "A5"
    p = fullfile(foot_db_root, ds, subset);
else
    p = fullfile(foot_db_root, ds);
end
end

function local_merge_people(subset_in)
if ~exist(subset_in, "dir")
    error("找不到目录: %s", subset_in);
end
dirs = dir(fullfile(subset_in, "P*"));
dirs = dirs([dirs.isdir]);
for i = 1:numel(dirs)
    person_name = string(dirs(i).name);
    person_dir = fullfile(subset_in, person_name);
    mats = dir(fullfile(person_dir, "*.mat"));
    if isempty(mats)
        continue;
    end
    data = [];
    for j = 1:numel(mats)
        fp = fullfile(person_dir, mats(j).name);
        loaded = load(fp);
        vars = fieldnames(loaded);
        if any(strcmp(vars, "geo_data"))
            x = loaded.geo_data;
        else
            x = loaded.(vars{1});
        end
        data = [data; x(:)];
    end
    geo_data = data;
    save(fullfile(subset_in, sprintf("%s_all.mat", person_name)), "geo_data");
end
end

function local_extract_events(subset_in, subset_out)
Fs = 8000;
tau = 1.2;
window = 0.35;
wndw_ovrlap = 0.40;
wndw_smpl = window * Fs;
sigma = 4.0;
cluster_num = 2;

all_mats = dir(fullfile(subset_in, "P*_all.mat"));
if isempty(all_mats)
    error("%s 下未找到任何 P*_all.mat", subset_in);
end
[~, idx] = sort(local_person_num({all_mats.name}));
all_mats = all_mats(idx);

seed_fp = fullfile(subset_in, all_mats(1).name);
seed_loaded = load(seed_fp);
geo_data = seed_loaded.geo_data;
n = min(numel(geo_data), 100 * Fs);
geo_data = smooth(geo_data(1:n), 5);
num_seg = floor(1 + (length(geo_data) - wndw_smpl) / (floor((1 - wndw_ovrlap) * wndw_smpl)));
signal_feat = [];
for i = 1:num_seg
    start = floor(wndw_smpl * (i - 1) * (1 - wndw_ovrlap) + 1);
    stop = floor(start + wndw_smpl - 1);
    if stop >= length(geo_data)
        stop = length(geo_data);
    end
    wght_wndw = length(start:stop);
    weight = gausswin(wght_wndw, tau);
    w_diag = diag(weight);
    sig = w_diag * geo_data(start:stop);
    signal_feat(i, :) = Events_Features_Extraction(Fs, sig);
end
[clust, cov_mat, mu_mat, phi] = GMM_EM(signal_feat, cluster_num);

if det(cov_mat(:, :, 1)) > det(cov_mat(:, :, 2))
    lbl_clst1 = 1;
    lbl_clst2 = 0;
else
    lbl_clst1 = 0;
    lbl_clst2 = 1;
end

persons = strings(1, numel(all_mats));
for i = 1:numel(all_mats)
    nm = string(all_mats(i).name);
    persons(i) = extractBefore(nm, "_all.mat");
end
person_names = cell(1, numel(persons));

for k = 1:numel(persons)
    person_names{k} = char(persons(k));
    fp = fullfile(subset_in, sprintf("%s_all.mat", persons(k)));
    loaded = load(fp);
    geo_data = smooth(loaded.geo_data, 5);
    num_seg = floor(1 + (length(geo_data) - wndw_smpl) / (floor((1 - wndw_ovrlap) * wndw_smpl)));
    Evnt_Ind = zeros(num_seg, 2);
    prob_clust = zeros(num_seg, 2);
    i = 1;
    iter = 1;
    while iter < length(geo_data)
        start = floor(wndw_smpl * (i - 1) * (1 - wndw_ovrlap) + 1);
        stop = floor(start + wndw_smpl - 1);
        if stop >= length(geo_data)
            stop = length(geo_data);
        end
        gaussian_wndw = length(geo_data(start:stop));
        weight = gausswin(gaussian_wndw, tau);
        w_diag = diag(weight);
        sig = w_diag * geo_data(start:stop);
        Evnt_Ind(i, :) = [start, stop];
        test_data = Events_Features_Extraction(Fs, sig);
        prob_clust(i, 1) = mvnpdf(test_data(:, 1:end), mu_mat(1, :), cov_mat(:, :, 1)) * phi(1);
        prob_clust(i, 2) = mvnpdf(test_data(:, 1:end), mu_mat(2, :), cov_mat(:, :, 2)) * phi(2);
        i = i + 1;
        iter = stop;
        if i > num_seg
            break;
        end
    end
    prob_clust = prob_clust ./ repmat(sum(prob_clust, 2), 1, size(prob_clust, 2));
    [c, clust_assign] = max(prob_clust, [], 2);
    id = find(c < 0.90);
    prdct_clst = zeros(size(clust_assign, 1), 1);
    c_idx = find(clust_assign == 1);
    prdct_clst(c_idx, :) = lbl_clst1;
    c_idx = find(clust_assign == 2);
    prdct_clst(c_idx, :) = lbl_clst2;
    prdct_clst(id, :) = 0;
    Evnt_Prdctd = [Evnt_Ind, prdct_clst];
    [~, footstep_feat] = Event_Extract(Evnt_Prdctd, geo_data, sigma, k);
    out_fp = fullfile(subset_out, sprintf("%s.mat", persons(k)));
    save(out_fp, "footstep_feat");
end
save(fullfile(subset_out, "person_names.mat"), "person_names");
end

function nums = local_person_num(names)
nums = zeros(1, numel(names));
for i = 1:numel(names)
    t = regexp(names{i}, "P(\d+)_all\.mat", "tokens", "once");
    if isempty(t)
        nums(i) = 10^9;
    else
        nums(i) = str2double(t{1});
    end
end
end
