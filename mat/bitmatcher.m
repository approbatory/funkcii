clear all
profile on
load ~/assets/c14m4d15/cm01-fix/rec_161217-182928.mat
lines = strsplit(fileread('/home/omer/assets/c14m4d15/cm01-fix/class_161217-183059.txt'), '\n');
goodcells = [];
for l=lines
    if ~isempty(l{1}) && isempty(strfind(l{1}, 'not'))
        goodcells = [goodcells, sscanf(l{1}, '%d')];
    end
end

traces = traces(:, goodcells);


%fprintf('sum of 123 is %f\n', sum(traces(:,ix)));

res = detect_events(traces);

py_tr = load('../ego-fix.tr');

if all(all(res.'==py_tr))
    disp('All match');
else
    disp('not all match');
end

profile viewer