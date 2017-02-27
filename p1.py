import math
"""
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):

    new = []
    count = -1
    
    slope_neg = 0
    neg_count = 0
    
    slope_pos = 0
    pos_count = 0
    
    neg_point = [0, 0]
    neg_length = 0
    
    pos_point = [0, 0]
    pos_length = 0
    
    high_point_pos = [0, 1000]
    low_point_pos = [1000, 0]
    
    high_point_neg = [1000, 1000]
    low_point_neg = [0, 0]
    
    #lines = sorted(lines, key=lambda x: -math.sqrt((x[0][3]-x[0][1])**2 + (x[0][2]-x[0][0])**2))
    #posi = [line for line in lines if (line[0][3]-line[0][1])/(line[0][2]-line[0][0]) > 0]
    #negi = [line for line in lines if (line[0][3]-line[0][1])/(line[0][2]-line[0][0]) < 0]
    
    
    
    #if len(posi) > 1:
    #    posi = posi[:int(len(posi)*0.5)]
        
    #if len(negi) > 1:
    #    negi = negi[:int(len(negi)*0.5)]
    #posi.extend(negi)
    #lines = posi
    
    #if len(lines) > 3:
    #    lines = lines[:int(len(lines)*0.9)]

    regress_pos = [[], [], []]
    regress_neg = [[], [], []]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            new.append([])
            count += 1
            slope = (y2-y1)/(x2-x1)
            
            
            length = math.sqrt((y2-y1)**2 + (x2-x1)**2)
            
            if slope < 0:
                regress_neg[0].append((x1))
                regress_neg[1].append((y1))
                regress_neg[0].append((x2))
                regress_neg[1].append((y2))
                regress_neg[2].append(int(length))
                regress_neg[2].append(int(length))
                
                neg_count += 1
                neg_length += length
                slope_neg = (slope_neg*(neg_length-length) + slope*length)/neg_length
                
                neg_point = [(neg_point[0]*(neg_length-length) + length*(x2+x1)/2)/neg_length, 
                            (neg_point[1]*(neg_length-length) + length*(y2+y1)/2)/neg_length]
                
            else:
                regress_pos[0].append((x1))
                regress_pos[1].append((y1))
                regress_pos[0].append((x2))
                regress_pos[1].append((y2))
                regress_pos[2].append(int(length))
                regress_pos[2].append(int(length))
                
                pos_count += 1
                pos_length += length
                slope_pos = (slope_pos*(pos_length-length) + slope*length)/pos_length
                pos_point = [((pos_point[0]*(pos_length-length) + length*(x2+x1)/2)/pos_length), 
                            ((pos_point[1]*(pos_length-length) + length*(y2+y1)/2)/pos_length)]
            
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)

                
#             for lin in lines:
#                 for x11,y11,x22,y22 in lin:
#                     new[count].append((y22-y11)/(x22-x11) - slope)
#             (new[count]).sort()
            
            
    #print(new)
    #print(slope_pos)
    #print(pos_point)
    #print(slope_neg)
    #print(neg_point)
    
    thickness = 10
    
    #x_pos_539 = int((539-pos_point[1]) /slope_pos + pos_point[0])
    #x_pos_330 = int((330-pos_point[1]) /slope_pos + pos_point[0])
    
    #x_neg_539 = int((539-neg_point[1]) /slope_neg + neg_point[0])
    #x_neg_330 = int((330-neg_point[1]) /slope_neg + neg_point[0])
    
    #cv2.line(img, (x_pos_539, 539), (x_pos_330, 330), color, thickness)
    #cv2.line(img, (x_neg_539, 539), (x_neg_330, 330), color, thickness)
    
    # regression trial
    print(regress_pos)
    
    x = np.array(regress_pos[0], dtype=np.float64)
    y = np.array(regress_pos[1], dtype=np.float64)
    w = np.array(regress_pos[2], dtype=np.float64)
    
    x = np.transpose(np.matrix(regress_pos[0]))
    y = np.transpose(np.matrix(regress_pos[1]))
    
    
    
    print(x.shape)
    print(y.shape)
    print(w.shape)
    
    x.reshape(x.shape[0],1)
    y.reshape(x.shape[0],1)
    w.reshape(x.shape[0],1)
    
    print(x.shape)
    print(y.shape)
    print(w.shape)
    
    print(x)
    print(y)
    print(w)
    
    
    
    model = LR()
    model.fit(x,y,w)
    print("slope: ", model.coef_)
    print("residues: ", model.residues_)
    model.predict(x)



"""

def dist(line):
        return math.sqrt((line[0]-line[2])**2 + (line[1]-line[3])**2)

def slope(line):
        if line[2] == line[0]:
            return None
        return (line[3]-line[1])/(line[2]-line[0])

def get_b(point, slope):
        if slope is None:
            return point[1]
        return point[1] - point[0]*slope

def get_y(x, slope, b):
        if slope is None:
            return b
        return x*slope + b

def get_xint(line):
    slp = slope(line)
    if slp is None:
        return line[0]
    if slp == 0:
        return None
    return -get_b(line, slp)/slp

def get_yint(line):
    slp = slope(line)
    if slp is None:
        return None
    return get_b(line, slp)

def get_midx(line):
    return (line[0] + line[2])/2

def get_midy(line):
    return (line[1] + line[3])/2


def create_vert_filter(x1, x2, ymx, left=True):
    """Filters lines to the left or right of the line x1, 0 and x2, ymx"""
    xmx = max(x1, x2)
    xmi = min(x1, x2)
    line = [x1, 0, x2, ymx]
    slp = slope(line)
    b = get_b(line, slp)
    if x1 < x2:
        below = True
    else:
        below = False
    
    def y(x, left):
        return get_y(x, slp, b) * 1 if left else -1

    def filt(line):
        nonlocal xmx, xmi, left
        line = line if left is True else list(map(lambda x: -x, line))
        _xmx, _xmi = (xmx, xmi) if left is True else (-xmx, -xmi)

        if line[0] > _xmx or line[2] > _xmx:
            return False

        if line[0] > _xmi:
            if y(line[0], left) > line[1]:
                 return False

        if line[2] > _xmi:
            if y(line[2], left) > line[3]:
                 return False
        return True

    return filt

def create_slope_filter(slpmi=None, slpmx=None):
    def filt(line):
        nonlocal slpmi, slpmx
        slp = slope(line)
        if not slp:
            return False
        if slpmi:
            if slp <= slpmi:
                return False
        if slpmx:
            if slp >= slpmx:
                return False
        return True

    return filt

def create_intercept_filter(xintmi=None, xintmx=None, yintmi=None, yintmx=None):
    inf = 1000000
    if not xintmi:
        xintmi = -inf
    if not xintmx:
        xintmx = inf
    if not yintmi: 
        yintmi = -inf
    if not yintmx:
        yintmx = inf

    def filt(line):
        nonlocal xintmi, xintmx, yintmi, yintmx
        xint = get_xint(line)
        yint = get_yint(line)
        if xintmi<=xint<=xintmx and yintmi<=yint<=yintmx:
            return True
        return False
    return filt

def create_filter_pipeline(*args):
    def filt(line):
        for arg in args:
            if arg(line) == False:
                return False
        return True
    return filt

def create_classify_pipeline(classes_dict):
    def classify(line):
        for _class in classes_dict:
            if classes_dict[_class](line):
                return _class
        return None
    return classify

def classify_lines(lines, classes_dict):
    classifier = create_classify_pipeline(classes_dict)
    res = {_class:[] for _class in classes_dict}
    res[None] = []

    for line in lines:
        res[classifier(line)].append(line)

    return res

def classified_lines_gen(lines, classes_dict):
    classifier = create_classify_pipeline(classes_dict)

    for line in lines:
        yield (classifier(line), line)


def transform_line_to_points(line, *args):
    """Format: slope, [pointx, pointy], length"""
    b = line[1][1]-line[0]*line[1][0]
    slp = line[0]
    res = []
    if slp is None:
        res = []
        for arg in args:
            res.append(line[1][0])
            res.append(arg)
        return res
    
    if slp == 0:
        res = []
        for arg in args:
            res.append(None)
            res.append(arg)
        return res

    for arg in args:
        res.append((arg-b)/slp)
        res.append(arg)
    return res


def merge_lines(lines, classes):
    """Format: slope, [pointx, pointy], length"""
    res = {_class:[0, [0, 0], 0] for _class in classes}

    for classified in classified_lines_gen(lines, classes):
        line = classified[1]
        _class = classified[0]

        if not _class is None:
            slp = slope(line)
            slp = 1000 if slp is None else slp
            res[_class][0] = (res[_class][0]*res[_class][2] + slp*dist(line))/ (dist(line) + res[_class][2])
            res[_class][1] = [
                                (res[_class][1][0]*res[_class][2] + get_midx(line)*dist(line))/ (dist(line) + res[_class][2]),
                                (res[_class][1][1]*res[_class][2] + get_midy(line)*dist(line))/ (dist(line) + res[_class][2])
            ]
            res[_class][2] += dist(line)

    return res


def avg_lines(lines):
    ymi = min([lines[0][1], lines[0][3]])
    ymx = max([lines[0][1], lines[0][3]])
    slp = 0
    b = 0
    count = 0
    for line in map(lambda line: (slope(line), get_b(line, slope(line))), lines):
        count += 1
        slp = (slp*(count-1) + line[0])/count
        b = (b*(count-1) + line[1])/count
    return [(ymi-b)/slp, ymi, (ymx-b)/slp, ymx]


def test_filter_pipeline(lines, img):
    #print(img.shape)
    image_shape = (img.shape[0], img.shape[1], 3)
    ymax = image_shape[0]

    ymx = image_shape[0] - 1
    ymid = image_shape[0]//2 + 120
    
    xmax = image_shape[1]
    xmid = image_shape[1]//2

    left_vert_filt =  create_filter_pipeline(create_vert_filter(xmid , xmid , ymax))
    left_slope_filt = create_slope_filter(-1, -0.5)
    left_int_filt = create_intercept_filter(xmax-200, xmax+200)

    left_filt = create_filter_pipeline(left_vert_filt, left_slope_filt, left_int_filt)

    right_vert_filt =  create_filter_pipeline(create_vert_filter(xmid , xmid, ymax, False))
    right_slope_filt = create_slope_filter(0.5, 1)
    right_int_filt = create_intercept_filter(-120, 200)

    right_filt = create_filter_pipeline(right_vert_filt, right_slope_filt, right_int_filt)
    
    classes = {"left": left_filt, "right": right_filt}
    res = frame_averager({_class:transform_line_to_points(line, ymx, ymid) 
                          for _class, line in merge_lines(lines, classes).items() }).values()
    #print(res)
    #res = [transform_line_to_points(line, ymx, ymid) for line in  res ]
    
    #for line in res:
    #    print("x intercept: @@@@ ", get_xint(line))
    
    return res
    


def average_frames(frame_sync=5):
    frames = []
    def avg(classes):
        nonlocal frames, frame_sync
        for _class in classes:
            lines = [classes[_class]]
            for frame in frames:
                lines.append(frame[_class])
            classes[_class] = avg_lines(lines)
        frames.append(classes)
        if len(frames) > frame_sync:
            del frames[0]
    return avg


def test():
    lines = [[10, 10, 20, 20],[10, 10, 20, 20],[10, 10, 20, 20],[10, 10, 20, 20],]
    print(avg_lines(lines))
    print("\n\n\n")


    image_shape = (540, 960, 3)
    ymax = image_shape[0]

    ymx = image_shape[0] - 1
    ymid = image_shape[0]/2

    xmid = image_shape[1]/2

    left_vert_filt =  create_filter_pipeline(create_vert_filter(xmid , xmid , ymax))

    left_slope_filt = create_slope_filter(-5, -0.577)

    left_int_filt = create_intercept_filter(xmid+50, 1000)



    left_filt = create_filter_pipeline(left_vert_filt, left_slope_filt, left_int_filt)





    right_vert_filt =  create_filter_pipeline(create_vert_filter(xmid , xmid, ymax, False))

    right_slope_filt = create_slope_filter(0.577, 5)

    right_int_filt = create_intercept_filter(-100, xmid-50)

    

    right_filt = create_filter_pipeline(right_vert_filt, right_slope_filt, right_int_filt)


    classes = {"left": left_filt, "right": right_filt}


    lines = [[[517, 331, 877, 538]],

             [[521 ,330 ,897, 538]],

             [[383 ,382 ,406 ,365]],

             [[281 ,460 ,345, 410]],

             [[295 ,461, 352 ,411]],

             [[389, 380, 416 ,355]],

             [[413 ,364 ,449 ,334]],

             [[411, 365, 440 ,339]],

             [[338 ,425, 352, 412]],

             [[383, 381 ,456 ,332]],

             [[383 ,380 ,437, 340]],

             [[520 ,330 ,656 ,405]],

             [[538 ,342 ,834, 513]],

             [[514 ,330, 532, 340]],

             [[281 ,461 ,305 ,442]],

             [[631, 390, 898, 538]],

             [[293,462 ,311 ,446]],

             [[847, 522, 876 ,539]],

             [[435, 342 ,450, 331]],

             [[388 ,382, 416, 360]]]

    lines = [x[0] for x in lines]

    print(classify_lines(lines, classes))
    print(merge_lines(lines, classes))
    print( [transform_line_to_points(line, ymx, ymid) for line in  merge_lines(lines, classes).values() ] )
    # for line in lines:
    #     print(slope(line), get_b(line, slope(line)), get_xint(line), get_yint(line))
    #     print(line)
    #     print(filt(line))

if __name__ == "__main__":
    test()

#@adaml So what sort of clustering algorithm did you write? I thought of something simple like 






































