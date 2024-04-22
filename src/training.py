

test_bbox = [60, 1197, 4680, 2288]
train_bbox = [60, 2288, 4680, 5857]
val_bbox = [60, 5857, 4680, 6441]


def read_points_coord(bbox,fpath):
    pass

im_path_ut = r""
im_path_notut = r""

train_points_ut = read_points_coord(train_bbox,im_path_ut)
train_points_notut = read_points_coord(train_bbox,im_path_notut)

val_points_ut = read_points_coord(val_bbox,im_path_ut)
val_points_notut = read_points_coord(val_bbox,im_path_notut)

test_points_ut = read_points_coord(test_bbox,im_path_ut)
test_points_notut = read_points_coord(test_bbox,im_path_notut)