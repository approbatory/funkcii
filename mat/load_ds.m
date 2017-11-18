function ds = load_ds(dir)
start_dir = pwd;
cd(dir);
sources = data_sources;
if exist('cm01-fix', 'dir')
    ds = DaySummary(sources, 'cm01-fix');
elseif exist('cm01', 'dir')
    ds = DaySummary(sources, 'cm01');
end
cd(start_dir);
end
