clc; clear all; close all;

% Ask user to select the root folder that contains P1–P100, or P1-P30, P1-P40 subfolders
rootFolder = uigetdir(pwd, 'Select the root folder containing P1–P5 subfolders');
if rootFolder == 0
    error('No folder selected. Exiting.');
end

% 获取 rootFolder 下所有以 P 开头的文件夹
dirInfo = dir(fullfile(rootFolder, 'P*'));
% 过滤不是文件夹的项目
dirInfo = dirInfo([dirInfo.isdir]);
persons = {dirInfo.name};

if isempty(persons)
    error('在选定目录中未找到任何以 P 开头的参与者文件夹 (如 P1, P2 等)。');
end

fprintf('找到 %d 个参与者文件夹。\n', length(persons));

dataset_original = [];

for j = 1:length(persons)
    personName = persons{j};
    personFolder = fullfile(rootFolder, personName);
    
    fprintf('正在处理参与者: %s...\n', personName);

    data = [];

    % 自动获取该文件夹下所有的 .mat 文件
    matFiles = dir(fullfile(personFolder, '*.mat'));
    if isempty(matFiles)
        warning('文件夹 %s 中没有找到 .mat 文件，跳过。', personName);
        continue;
    end

    for i = 1:length(matFiles)
        filepath = fullfile(personFolder, matFiles(i).name);
        
        try
            loadedData = load(filepath);
            % 假设变量名是 geo_data，如果不是，尝试获取第一个变量
            vars = fieldnames(loadedData);
            if ismember('geo_data', vars)
                tempdata = loadedData.geo_data;
            else
                tempdata = loadedData.(vars{1});
            end
            data = [data; tempdata];
        catch ME
            warning('加载文件 %s 出错: %s', filepath, ME.message);
        end
    end

    if ~isempty(data)
        % Save combined data in root folder
        geo_data = data;
        dataname = sprintf('%s_all.mat', personName);
        save(fullfile(rootFolder, dataname), 'geo_data');
        fprintf('  已保存: %s\n', dataname);
    else
        warning('参与者 %s 没有有效数据，未生成合并文件。', personName);
    end
end
