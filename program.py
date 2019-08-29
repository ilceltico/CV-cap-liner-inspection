import utils
import circledetection
import cv2
import numpy as np

def outer_circle_detection():
    pass

def inner_circle_detection():
    pass

def liner_defects_detection():
    pass

def execute():
    missing_liner_threshold = utils.get_missing_liner_threshold()
    print('Missing liner threshold: ' + str(missing_liner_threshold))
    
    for file in os.listdir('./caps'):
        print('--------------------------------------------------')
        print(file)

        img = cv2.imread('caps/' + file, cv2.IMREAD_GRAYSCALE)
        binary = utils.binarize(img)
        #cv2.imshow('binary', binary)

        # test if the cap is a circle
        if not utils.is_circle(binary):
            print('The cap in ' + file + ' is NOT a circle')
            continue
        else:
            print('The cap in ' + file + ' is a circle')

        #we need to convert it to a boolean mask (as linerdefects_gradient.circularmask does: it creates a circular boolean mask)
        mask = binary.copy().astype(bool)

        # TASK1
        print('TASK1')

        x_cap, y_cap, r_cap = outer_circle_detection()

        if not (x_cap is None or y_cap is None or r_cap is None):
            print('Position of the center of the cap: (' + str(x_cap) + ', ' + str(y_cap) + ')')
            print('Diameter of the cap: ' + str(2 * r_cap))

            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(y_cap).astype('int'), np.round(x_cap).astype('int')), np.round(r_cap).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(y_cap).astype('int'), np.round(x_cap).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' outer circle (cap)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print('Is the liner missing? ')
            #mask = utils.circular_mask(img.shape[0], img.shape[1], (circle[0], circle[1]), r_cap)
            avg = np.mean(img[mask])
            print('caps/' + file + ' pixels average: ' + str(avg))

            if avg > missing_liner_threshold:
                print('caps/' + file + ' has NO liner')
                continue
            else:
                print('caps/' + file + ' has liner')

        # TASK2
        print('TASK2')

        x_liner, y_liner, r_liner = inner_circle_detection()

        if not (x_liner is None or y_liner is None or r_liner is None):
            print('Position of the center of the liner: (' + str(x_liner) + ', ' + str(y_liner) + ')')
            print('Diameter of the liner: ' + str(2 * r_liner))

            coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cv2.circle(coloured_image, (np.round(y_liner).astype('int'), np.round(x_liner).astype('int')), np.round(r_liner).astype('int'), (0, 255, 0), 1)
            cv2.circle(coloured_image, (np.round(y_liner).astype('int'), np.round(x_liner).astype('int')), 2, (0, 0, 255), 3)
            cv2.imshow('caps/' + file + ' inner circle (liner)', coloured_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # DEFECT DETECTION
            print('Is the liner incomplete?')

            has_defects, rectangle = liner_defects_detection()

            if not has_defects :
                print('caps/' + file + ' has NO defects')
            else:
                print('caps/' + file + ' has defects')
                coloured_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawContours(coloured_image, [rectangle], 0, (0,0,255), 1)
                cv2.imshow('caps/' + file + ' detected defects', coloured_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()


if __name__ == '__main__':
    execute()
