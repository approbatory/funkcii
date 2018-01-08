% takes the full set of traces and iteratively finds a threshold value for
% each trace by removing zscore > z items, recalculating the threshold
% values, removing, etc. until a fixed point is reached or maximum number
% of iterations

function thresh = peakfree(trace, z, max_iters)
for it=1:max_iters
    thresh = mean(trace, 'omitnan') + z*std(trace, 'omitnan');
    filt = (trace > thresh) & ~isnan(trace);
    if ~any(any(filt))
        fprintf('stopped on iter %d\n', it);
        return;
    end
    trace(filt) = nan;
end
fprintf('stopped at end (iter %d)\n', it);
end