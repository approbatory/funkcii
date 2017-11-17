clear;
ds = load_ds('~/brain/hpc/assets/c14m4d15/');
%% load xy position data
n_trials = length(ds.trials);

pos = cell(n_trials,1);
for i = 1:n_trials
    pos{i} = ds.trials(i).centroids * [1 1;1 -1]/sqrt(2);
end
all_pos = cell2mat(pos);

mins = min(all_pos);
maxs = max(all_pos);

%all_pos_resc = (all_pos - mins)./(maxs - mins);
for i = 1:n_trials
    pos{i} = (pos{i} - mins)./(maxs-mins);
end
clear all_pos mins maxs i
%% example trial view
figure;
c = pos{1};
plot(c(:,1), c(:,2), '.');
hold on
plot(c(70,1), c(70,2), 'rx');
xlim([0 1]);
ylim([0 1]);
clear c
%% load labels and events
end_labels = {ds.trials.end};
label_isnorth = strcmp('north', end_labels);
evs = {ds.trials.events};

%position_on_start_arm = 0.3;
position_on_start_arm = 0.1;
frame_of_interest = zeros(n_trials,1);
for i = 1:n_trials
    tr_pos = pos{i}(:,1);
    if tr_pos(1) > 0.5
        tr_pos = 1-tr_pos;
    end
    frame_of_interest(i) = find(tr_pos > position_on_start_arm, 1);
end
clear i
figure
for i = 1:n_trials
    c = pos{i};
    plot(c(:,1), c(:,2), 'b.');
    hold on
    plot(c(frame_of_interest(i),1), c(frame_of_interest(i),2), 'rx');
    xlim([0 1]);
    ylim([0 1]);
end
%% TODO: generate fake multinomial vector at the frame of interest using the events detected, for each trial
%   Then using multinomial naive bayes decode the final arm using the
%   label. Divide the training data into test train 70/30, or later into
%   k-fold xval
X = gen_X_at_frames(evs, frame_of_interest);
label_to_class = containers.Map({'north','south','east','west'},{1,2,3,4});
class_to_label = containers.Map({1,2,3,4},{'north','south','east','west'});
ks = zeros(1,n_trials);
K = 4;
for i=1:n_trials
    ks(i) = label_to_class(ds.trials(i).end);
end
cutoff = 77;
X_train = X(:,1:cutoff);
ks_train = ks(1:cutoff);
X_test = X(:,cutoff+1:end);
ks_test = ks(cutoff+1:end);

[log_prior, log_conditional] = multinom_nb_encode(X_train, ks_train, K);
ks_predicted = multinom_nb_decode(X_test, log_prior, log_conditional);
mistakes = sum(ks_test ~= ks_predicted);
total = length(ks_test);
fprintf('Multinomial naive bayes had an error rate of %f, or %d out of %d\n',...
    mistakes/total, mistakes, total);
%TODO implement leave one out cross validation, and see if any of them turn
%out to be mistakes!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#@#@#@#@#@#@#@#@#
%%
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

function X = gen_X_at_frames(evs, frames)
assert(length(evs)==length(frames));
n_trials = length(evs);
n_cells = length(evs{1});
X = zeros(n_cells ,n_trials);
for i = 1:n_trials
    for j = 1:n_cells
        for e = evs{i}{j}'
            if frames(i) >= e(1) && frames(i) <= e(2)
                X(j,i) = e(3);
            end
        end
    end
end
end