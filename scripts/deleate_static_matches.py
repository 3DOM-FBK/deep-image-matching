STATIC_THRESHOLD_X_MIN = 0 # 88pixel
STATIC_THRESHOLD_X_MAX = 30 # 13pixel
STATIC_THRESHOLD_Y_MIN = 50 # 88pixel
STATIC_THRESHOLD_Y_MAX = 200 # 150pixel

WORK_DIR = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon_plastic_bottle_no_GT\SIFT_RGB_upright_P0.0066"

matches_file = r"{}\matches\matches.txt".format(WORK_DIR)
kpts_dir = r"{}\colmap_desc".format(WORK_DIR)
out_dir = r"{}\no_static".format(WORK_DIR)

def read_matches_txt(path_to_matches, matches_dict):
    with open(path_to_matches, 'r') as matches_file:
        lines = matches_file.readlines()
        matches_list = []
        img1 = None
        img2 = None
        for line in lines:
            if line != "\n":
                line = line.strip()
                element1, element2 = line.split(" ", 1)
                try:
                    match1 = int(element1)
                    match2 = int(element2)
                    matches_list.append((match1, match2))
                    matches_dict[(img1, img2)] = matches_list
                except:
                    img1 = element1
                    img2 = element2
                    matches_list = []
            elif line == "\n":
                print("Found empty line, it is not an error.")
    return matches_dict

kpt1_dict = {}
kpt2_dict = {}
matches_dict = {}

matches_dict = read_matches_txt(matches_file, matches_dict)
for key in matches_dict:
    img1 = key[0]
    img2 = key[1]
    #print(img1, img2, len(matches_dict[key]))
    #print(matches_dict[key]); quit()

    with open("{}/{}.txt".format(kpts_dir, img1), 'r') as kp1_file, open("{}/{}.txt".format(kpts_dir, img2), 'r') as kp2_file, open("{}\matches_no_static.txt".format(out_dir), 'a') as out_matches_file:
        kp1_lines = kp1_file.readlines()
        kp2_lines = kp2_file.readlines()

        for c1, line1 in enumerate(kp1_lines[1:]):
            x1, y1, _ = line1.split(" ", 2)
            kpt1_dict[c1] = [x1, y1]

        for c2, line2 in enumerate(kp2_lines[1:]):
            x2, y2, _ = line2.split(" ", 2)
            kpt2_dict[c2] = [x2, y2]

        #print(c1, c2)
        out_matches_file.write("{} {}\n".format(img1, img2))
  
        for item in matches_dict[key]:
            m1, m2 = item[0], item[1]
            m1 = int(m1)
            m2 = int(m2)
            #print(f"m1 m2 {m1} {m2}")
            x1 = kpt1_dict[m1][0]
            y1 = kpt1_dict[m1][1]
            x2 = kpt2_dict[m2][0]
            y2 = kpt2_dict[m2][1]

            #flux = ((float(x2)-float(x1))**2 + (float(y2)-float(y1))**2)**0.5
            flux_x = abs(float(x2) - float(x1))
            flux_y = abs(float(y2) - float(y1))
            #print(flux_x, flux_y)

            #if flux > STATIC_THRESHOLD:
            if flux_x > STATIC_THRESHOLD_X_MIN and flux_x < STATIC_THRESHOLD_X_MAX and flux_y > STATIC_THRESHOLD_Y_MIN and flux_y < STATIC_THRESHOLD_Y_MAX:
                #print('ok')
                out_matches_file.write("{} {}\n".format(m1, m2))

        out_matches_file.write("\n\n")



# For a single pair
#STATIC_THRESHOLD_X_MIN = 0 # 88pixel 
#STATIC_THRESHOLD_X_MAX = 30 # 13pixel
#STATIC_THRESHOLD_Y_MIN = 50 # 88pixel
#STATIC_THRESHOLD_Y_MAX = 200 # 150pixel
#
#matches_file = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\resized\matches\matches.txt"
#kpts_dir = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\resized\colmap_desc"
#out_dir = r"G:\My Drive\O3DM_2022\FinalPaper\pairs\nikon\PlasticBottle\resized\no_static"
#
#kpt1_dict = {}
#kpt2_dict = {}
#
#with open("{}".format(matches_file), "r") as matches_file:
#    matches_lines = matches_file.readlines()
#    img1, img2 = matches_lines[0].strip().split(" ", 1)
#    print(img1, img2)
#
#    with open("{}/{}.txt".format(kpts_dir, img1), 'r') as kp1_file, open("{}/{}.txt".format(kpts_dir, img2), 'r') as kp2_file, open("{}\matches_no_static.txt".format#(out_dir), 'w') as out_matches_file:
#        kp1_lines = kp1_file.readlines()
#        kp2_lines = kp2_file.readlines()
#
#        for c1, line1 in enumerate(kp1_lines[1:]):
#            x1, y1, _ = line1.split(" ", 2)
#            kpt1_dict[c1] = [x1, y1]
#
#        for c2, line2 in enumerate(kp2_lines[1:]):
#            x2, y2, _ = line2.split(" ", 2)
#            kpt2_dict[c2] = [x2, y2]
#
#        #print(c1, c2)
#        out_matches_file.write("{} {}\n".format(img1, img2))
#  
#        for match_line in matches_lines[1:-3]:
#            m1, m2 = match_line.split(" ", 1)
#            m1 = int(m1)
#            m2 = int(m2)
#            x1 = kpt1_dict[m1][0]
#            y1 = kpt1_dict[m1][1]
#            x2 = kpt2_dict[m2][0]
#            y2 = kpt2_dict[m2][1]
#
#            #flux = ((float(x2)-float(x1))**2 + (float(y2)-float(y1))**2)**0.5
#            flux_x = abs(float(x2) - float(x1))
#            flux_y = abs(float(y2) - float(y1))
#            #print(flux_x, flux_y)
#
#            #if flux > STATIC_THRESHOLD:
#            if flux_x > STATIC_THRESHOLD_X_MIN and flux_x < STATIC_THRESHOLD_X_MAX and flux_y > STATIC_THRESHOLD_Y_MIN and flux_y < STATIC_THRESHOLD_Y_MAX:
#                print('ok')
#                out_matches_file.write("{} {}\n".format(m1, m2))