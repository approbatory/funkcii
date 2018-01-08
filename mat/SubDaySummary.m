classdef SubDaySummary
    properties
        trials
        num_cells
        num_trials
        trial_indices
        full_num_frames
    end
    methods
        function obj = SubDaySummary(val)
            obj.trials = val.trials;
            obj.num_cells = val.num_cells;
            obj.num_trials = val.num_trials;
            obj.trial_indices = val.trial_indices;
            obj.full_num_frames = val.full_num_frames;
            if isfield(val, 'cells')
                obj.cells = val.cells;
            end
        end
    end
end
