function [ transient_canvas, outp ] = detect_events( traces )
%DETECT_EVENTS detects transient events and outputs binary values for the
%presence or absence of a transient
params.med_window_size = 101;
params.avg_window_size = 3;
params.z = 1.5;
params.width = 5;
%params.sep = 3;
%effective sep=4
params.sep = 4;
params.offset_delay = 3;

%subtract median
transients_sub_med = traces - medfilt1(traces, params.med_window_size, [], 2);

%sliding average
kern = ones(1,params.avg_window_size);
transients_cleaned = conv2(transients_sub_med, kern, 'same');
transients_cleaned = transients_cleaned/params.avg_window_size;

%maximum detection
threshold = params.z*std(transients_cleaned, 1, 2);
%passing_threshold = transients_cleaned >= threshold;
passing_threshold = bsxfun(@(a,b) a>=b, transients_cleaned, threshold);
kern = ones(1, params.width);
%repeated passing condition
repeated_passing = conv2(1.0*passing_threshold, 1.0*kern, 'same') >= params.width;
[rows, ~] = size(transients_cleaned);
local_maximum = [zeros(rows, 1), diff(diff(transients_cleaned, 1, 2) > 0, 1, 2) < 0, zeros(rows, 1)];
well_sep_max = local_maximum;
%replicate the bug from the python code whereby well_sep_max is not a copy
%of local_maximum, but the same object. So local maxima cannot invalidate a
%local maximum if they have been invalidated themselves. genius
for i = 1:params.sep
    %well_sep_max = well_sep_max & ~circshift(local_maximum, [0,i]);
    well_sep_max = well_sep_max & ~circshift(well_sep_max, [0,i]);
end
valid_transient_peaks = repeated_passing & well_sep_max;

%transient expansion
[rows, cols] = find(valid_transient_peaks~=0);
num_peaks = length(cols);
transient_canvas = zeros(size(valid_transient_peaks));
for ix = 1:num_peaks
    i = rows(ix);
    j = cols(ix);
    dropping = 1;
    old_val = transients_cleaned(i,j);
    while dropping && j > 1
        j = j-1;
        new_val = transients_cleaned(i,j);
        if new_val > old_val
            dropping = 0;
        end
        old_val = new_val;
    end
    %start_of_transient = max([j - params.offset_delay,1]);
    %end_of_transient = max([cols(ix) - params.offset_delay,1]);
    %TODO check this change
    start_of_transient = j - params.offset_delay;
    end_of_transient = cols(ix) - params.offset_delay;
    if (start_of_transient >= 1) && (end_of_transient >= 1)
        transient_canvas(i, start_of_transient:end_of_transient-1) = 1;
    end
end
outp{1} = traces;
outp{2} = transients_sub_med;
outp{3} = transients_cleaned;
outp{4} = valid_transient_peaks;
outp{5} = transient_canvas;
end