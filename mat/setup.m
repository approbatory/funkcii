cd ~/assets/c14m4d15/
sources = data_sources;
ds = DaySummary(sources, 'cm01-fix');
%de = @(x) detect_events(ds, x, 'fps', 10);
%events = de(46);
browse_rasters_touch(ds);