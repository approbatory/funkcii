clear;
ds = load_ds('~/brain/hpc/assets/c14m4d15/');
de_probe(ds);
%%
[poss, err, err_map] = decode_end_nb(ds, 0.005, 0.4);
view_err(ds, poss, err, err_map);
