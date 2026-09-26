import cv2


class HandDrawer:

    def __init__(self):

        self.fingers = {

            "thumb": {
                "points": [0,1,2,3,4],
                "color": (0,0,255)
            },

            "index": {
                "points": [0,5,6,7,8],
                "color": (255,120,0)
            },

            "middle": {
                "points": [0,9,10,11,12],
                "color": (0,255,0)
            },

            "ring": {
                "points": [0,13,14,15,16],
                "color": (0,255,255)
            },

            "pinky": {
                "points": [0,17,18,19,20],
                "color": (255,0,255)
            }

        }

    def draw(self, frame, landmarks):

        if len(landmarks) != 21:
            return

        # garis
        for finger in self.fingers.values():

            pts = finger["points"]
            color = finger["color"]

            for i in range(len(pts)-1):

                p1 = landmarks[pts[i]]
                p2 = landmarks[pts[i+1]]

                cv2.line(
                    frame,
                    (p1[1], p1[2]),
                    (p2[1], p2[2]),
                    color,
                    4
                )

        # joint putih
        for _, x, y in landmarks:

            cv2.circle(
                frame,
                (x,y),
                8,
                (255,255,255),
                -1
            )

            cv2.circle(
                frame,
                (x,y),
                10,
                (120,120,120),
                2
            )