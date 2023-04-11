import cv2
import dlib
import numpy as np
from flask import Flask, render_template, Response, request, redirect, url_for
import time
from itertools import count
import random

app = Flask(__name__)

RESIZE_HEIGHT = 360
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


@app.route('/')
def index():
    time.sleep(1.5)
    return render_template('index.html')


@app.route('/rect_video_feed', methods=['GET', 'POST'])
def rect_video_feed():
    # set the initial video feed to the default camera
    cap = cv2.VideoCapture(0)

    def gen_frames():
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(
                frame,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                det = max(detector(frame), key=lambda r: r.area())
            except Exception as e:
                print(e)
            else:
                cv2.rectangle(
                    frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255), 2
                )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/detect_face")
def detect_face():
    time.sleep(1.5)
    return render_template("detect_face.html")


@app.route("/detect_face_feed", methods=['GET', 'POST'])
def detect_face_feed():
    # set the initial video feed to the default camera
    cap = cv2.VideoCapture(0)

    img1 = cv2.imread("image9.jpeg")
    height, width = img1.shape[:2]
    IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
    img1 = cv2.resize(
        img1,
        None,
        fx=1.0 / IMAGE_RESIZE,
        fy=1.0 / IMAGE_RESIZE,
        interpolation=cv2.INTER_LINEAR,
    )

    det = max(detector(img1), key=lambda r: r.area())
    img1_face = img1[det.top(): det.bottom(), det.left(): det.right()]

    def gen_frames():
        while True:
            ret, img2 = cap.read()
            if not ret:
                continue

            height, width = img2.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            img2 = cv2.resize(
                img2,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                det = max(detector(img2), key=lambda r: r.area())
            except Exception as e:
                print(e)
            else:
                img2[det.top(): det.bottom(), det.left(): det.right()] = cv2.resize(
                    img1_face, (det.right() - det.left(), det.bottom() - det.top())
                )

            frame = cv2.imencode('.jpg', img2)[1].tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/face_alignment")
def face_alignment():
    time.sleep(1.5)
    return render_template("face_alignment.html")


@app.route("/face_alignment_feed", methods=['GET', 'POST'])
def face_alignment_feed():
    # set the initial video feed to the default camera
    cap = cv2.VideoCapture(0)

    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def gen_frames():
        while True:
            ret, img = cap.read()
            if not ret:
                continue

            height, width = img.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            img = cv2.resize(
                img,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                points = detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO)
            except Exception as e:
                print(e)
            else:
                for point in points:
                    cv2.circle(img, point, 2, (0, 0, 255), -1)

            # convert the frame to JPEG format
            ret, buffer = cv2.imencode(".jpg", img)
            frame = buffer.tobytes()

            # yield the frame in byte format
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/find_convex_hull")
def find_convex_hull():
    time.sleep(1.5)
    return render_template("find_convex_hull.html")


@app.route("/find_convex_hull_feed", methods=['GET', 'POST'])
def find_convex_hull_feed():
    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def gen_frames():
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            if not ret:
                continue

            height, width = img.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            img = cv2.resize(
                img,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )

            try:
                points = detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO)
            except Exception as e:
                print(e)
            else:
                hull_points = cv2.convexHull(np.array(points))
                img = cv2.fillPoly(img, [hull_points], (255, 0, 0))

            ret, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/affine_wrap_triangles")
def affine_wrap_triangles():
    time.sleep(1.5)
    return render_template("affine_wrap_triangles.html")


@app.route("/affine_wrap_triangles_feed", methods=['GET', 'POST'])
def affine_wrap_triangles_feed():
    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def get_delaunay_triangles(rect, points, indexes):
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)

        found_triangles = subdiv.getTriangleList()

        delaunay_triangles = []

        def contains(rect, point):
            return (
                    rect[0] < point[0] < rect[0] + rect[2]
                    and rect[1] < point[1] < rect[1] + rect[3]
            )

        for t in found_triangles:
            triangle = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

            # `getTriangleList` return triangles only, without origin points indices and we need them
            # so they correspond to other picture through index. So we're looking for original
            # index number for every point.
            if (
                    contains(rect, triangle[0])
                    and contains(rect, triangle[1])
                    and contains(rect, triangle[2])
            ):

                indices = []
                for index, point in enumerate(points):
                    if (
                            triangle[0][0] == point[0]
                            and triangle[0][1] == point[1]
                            or triangle[1][0] == point[0]
                            and triangle[1][1] == point[1]
                            or triangle[2][0] == point[0]
                            and triangle[2][1] == point[1]
                    ):
                        indices.append(indexes[index])

                    if len(indices) == 3:
                        delaunay_triangles.append(indices)
                        continue
        # remove duplicates
        return list(set(tuple(t) for t in delaunay_triangles))

    def warp_triangle(img1, img2, t1, t2):
        # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
        bb1 = cv2.boundingRect(np.float32([t1]))

        img1_cropped = img1[bb1[1]: bb1[1] + bb1[3], bb1[0]: bb1[0] + bb1[2]]

        bb2 = cv2.boundingRect(np.float32([t2]))

        t1_offset = [
            ((t1[0][0] - bb1[0]), (t1[0][1] - bb1[1])),
            ((t1[1][0] - bb1[0]), (t1[1][1] - bb1[1])),
            ((t1[2][0] - bb1[0]), (t1[2][1] - bb1[1])),
        ]
        t2_offset = [
            ((t2[0][0] - bb2[0]), (t2[0][1] - bb2[1])),
            ((t2[1][0] - bb2[0]), (t2[1][1] - bb2[1])),
            ((t2[2][0] - bb2[0]), (t2[2][1] - bb2[1])),
        ]
        mask = np.zeros((bb2[3], bb2[2], 3), dtype=np.float32)

        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), cv2.LINE_AA)

        size = (bb2[2], bb2[3])

        mat = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

        img2_cropped = cv2.warpAffine(
            img1_cropped,
            mat,
            (size[0], size[1]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        img2_cropped = img2_cropped * mask

        img2_cropped_slice = np.index_exp[
                             bb2[1]: bb2[1] + bb2[3], bb2[0]: bb2[0] + bb2[2]
                             ]
        img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
        img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped

    def gen_frames():
        img1 = cv2.imread("image4.jpeg")
        height, width = img1.shape[:2]
        RESIZE_HEIGHT = 500
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        img1 = cv2.resize(
            img1,
            None,
            fx=1.0 / IMAGE_RESIZE,
            fy=1.0 / IMAGE_RESIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        points1 = detect_facial_landmarks(img1, FACE_DOWNSAMPLE_RATIO)

        hull_index = cv2.convexHull(np.array(points1), returnPoints=False)

        # Mouth points
        # https://www.pyimagesearch.com/wp-content/uploads/2017/04/facial_landmarks_68markup-768x619.jpg
        mouth_points = [
            [60],  # <inner mouth>
            [61],
            [62],
            [63],
            [64],
            [65],
            [66],
            [67],  # </inner mouth>
        ]

        hull_index = np.concatenate((hull_index, mouth_points))
        hull1 = [points1[hull_index_element[0]] for hull_index_element in hull_index]

        rect = (0, 0, img1.shape[1], img1.shape[0])
        delaunay_triangles = get_delaunay_triangles(rect, hull1, [hi[0] for hi in hull_index])

        cap = cv2.VideoCapture(0)

        for i in count(1):
            ret, img2 = cap.read()
            if not ret:
                continue

            height, width = img2.shape[:2]
            RESIZE_HEIGHT = 500
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            img2 = cv2.resize(
                img2,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                points2 = detect_facial_landmarks(img2, FACE_DOWNSAMPLE_RATIO)
            except Exception as e:
                print(e)
            else:
                hull2 = [points2[hull_index_element[0]] for hull_index_element in hull_index]

                img1_warped = np.float32(img2)

                for triangle in delaunay_triangles:
                    mouth_points_set = set(mp[0] for mp in mouth_points)
                    if (
                            triangle[0] in mouth_points_set
                            and triangle[1] in mouth_points_set
                            and triangle[2] in mouth_points_set
                    ):
                        continue
                    t1 = [points1[triangle[0]], points1[triangle[1]], points1[triangle[2]]]
                    t2 = [points2[triangle[0]], points2[triangle[1]], points2[triangle[2]]]

                    warp_triangle(img1, img1_warped, t1, t2)

                img2 = np.uint8(img1_warped)

            ret, buffer = cv2.imencode(".jpg", img2)
            frame = buffer.tobytes()
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/delaunay_triangulation")
def delaunay_triangulation():
    time.sleep(1.5)
    return render_template("delaunay_triangulation.html")


@app.route("/delaunay_triangulation_feed", methods=["GET", "POST"])
def delaunay_triangulation_feed():
    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def get_delaunay_triangles(rect, points, indexes):
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)

        found_triangles = subdiv.getTriangleList()

        delaunay_triangles = []

        def contains(rect, point):
            return (
                    rect[0] < point[0] < rect[0] + rect[2]
                    and rect[1] < point[1] < rect[1] + rect[3]
            )

        for t in found_triangles:
            triangle = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

            # `getTriangleList` return triangles only, without origin points indices and we need them
            # so they correspond to other picture through index. So we're looking for original
            # index number for every point.
            if (
                    contains(rect, triangle[0])
                    and contains(rect, triangle[1])
                    and contains(rect, triangle[2])
            ):

                indices = []
                for index, point in enumerate(points):
                    if (
                            triangle[0][0] == point[0]
                            and triangle[0][1] == point[1]
                            or triangle[1][0] == point[0]
                            and triangle[1][1] == point[1]
                            or triangle[2][0] == point[0]
                            and triangle[2][1] == point[1]
                    ):
                        indices.append(indexes[index])

                    if len(indices) == 3:
                        delaunay_triangles.append(indices)
                        continue

        # remove duplicates
        return list(set(tuple(t) for t in delaunay_triangles))

    def gen_frames():
        COLORS = []
        first_frame = True

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(
                frame,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                points = detect_facial_landmarks(frame, FACE_DOWNSAMPLE_RATIO)
            except Exception:
                pass
            else:
                if first_frame:
                    hull_index = cv2.convexHull(np.array(points), returnPoints=False)
                    mouth_points = [

                        # [48],  # <outer mouth>
                        # [49],
                        # [50],
                        # [51],
                        # [52],
                        # [53],
                        # [54],
                        # [55],
                        # [56],
                        # [57],
                        # [58],  # </outer mouth>
                        [60],  # <inner mouth>
                        [61],
                        [62],
                        [63],
                        [64],
                        [65],
                        [66],
                        [67],  # </inner mouth>
                    ]
                    hull_index = np.concatenate((hull_index, mouth_points))
                    hull = [points[hull_index_element[0]] for hull_index_element in hull_index]

                    mouth_points_set = set(mp[0] for mp in mouth_points)

                    rect = (0, 0, frame.shape[1], frame.shape[0])
                    delaunay_triangles = get_delaunay_triangles(
                        rect, hull, [hi[0] for hi in hull_index]
                    )
                    # remove mouth points:
                    delaunay_triangles[:] = [
                        dt
                        for dt in delaunay_triangles
                        if not (
                                dt[0] in mouth_points_set
                                and dt[1] in mouth_points_set
                                and dt[2] in mouth_points_set
                        )
                    ]

                    COLORS[:] = [
                        (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
                        for _ in delaunay_triangles
                    ]
                    first_frame = False

                for color, triangle in zip(COLORS, delaunay_triangles):
                    frame = cv2.fillPoly(
                        frame, [np.array([points[index] for index in triangle])], color
                    )

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/seamless_cloning")
def seamless_cloning():
    time.sleep(1.5)
    return render_template("seamless_cloning.html")


@app.route("/seamless_cloning_feed", methods=["GET", "POST"])
def seamless_cloning_feed():
    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def get_delaunay_triangles(rect, points, indexes):
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)

        found_triangles = subdiv.getTriangleList()

        delaunay_triangles = []

        def contains(rect, point):
            return (
                    rect[0] < point[0] < rect[0] + rect[2]
                    and rect[1] < point[1] < rect[1] + rect[3]
            )

        for t in found_triangles:
            triangle = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

            # `getTriangleList` return triangles only, without origin points indices and we need them
            # so they correspond to other picture through index. So we're looking for original
            # index number for every point.
            if (
                    contains(rect, triangle[0])
                    and contains(rect, triangle[1])
                    and contains(rect, triangle[2])
            ):

                indices = []
                for index, point in enumerate(points):
                    if (
                            triangle[0][0] == point[0]
                            and triangle[0][1] == point[1]
                            or triangle[1][0] == point[0]
                            and triangle[1][1] == point[1]
                            or triangle[2][0] == point[0]
                            and triangle[2][1] == point[1]
                    ):
                        indices.append(indexes[index])

                    if len(indices) == 3:
                        delaunay_triangles.append(indices)
                        continue

        # remove duplicates
        return list(set(tuple(t) for t in delaunay_triangles))

    def warp_triangle(img1, img2, t1, t2):
        # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
        bb1 = cv2.boundingRect(np.float32([t1]))

        img1_cropped = img1[bb1[1]: bb1[1] + bb1[3], bb1[0]: bb1[0] + bb1[2]]

        bb2 = cv2.boundingRect(np.float32([t2]))

        t1_offset = [
            ((t1[0][0] - bb1[0]), (t1[0][1] - bb1[1])),
            ((t1[1][0] - bb1[0]), (t1[1][1] - bb1[1])),
            ((t1[2][0] - bb1[0]), (t1[2][1] - bb1[1])),
        ]
        t2_offset = [
            ((t2[0][0] - bb2[0]), (t2[0][1] - bb2[1])),
            ((t2[1][0] - bb2[0]), (t2[1][1] - bb2[1])),
            ((t2[2][0] - bb2[0]), (t2[2][1] - bb2[1])),
        ]
        mask = np.zeros((bb2[3], bb2[2], 3), dtype=np.float32)

        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), cv2.LINE_AA)

        size = (bb2[2], bb2[3])

        mat = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

        img2_cropped = cv2.warpAffine(
            img1_cropped,
            mat,
            (size[0], size[1]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        img2_cropped = img2_cropped * mask

        img2_cropped_slice = np.index_exp[
                             bb2[1]: bb2[1] + bb2[3], bb2[0]: bb2[0] + bb2[2]
                             ]
        img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
        img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped

    def gen_frames():
        img1 = cv2.imread("image9.jpeg")
        height, width = img1.shape[:2]
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        img1 = cv2.resize(
            img1,
            None,
            fx=1.0 / IMAGE_RESIZE,
            fy=1.0 / IMAGE_RESIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        points1 = detect_facial_landmarks(img1, FACE_DOWNSAMPLE_RATIO)

        original_hull_index = cv2.convexHull(np.array(points1), returnPoints=False)
        mouth_points = [
            # [48],  # <outer mouth>
            # [49],
            # [50],
            # [51],
            # [52],
            # [53],
            # [54],
            # [55],
            # [56],
            # [57],
            # [58],  # </outer mouth>
            [60],  # <inner mouth>
            [61],
            [62],
            [63],
            [64],
            [65],
            [66],
            [67],  # </inner mouth>
        ]

        hull_index = np.concatenate((original_hull_index, mouth_points))
        hull1 = [points1[hull_index_element[0]] for hull_index_element in hull_index]

        rect = (0, 0, img1.shape[1], img1.shape[0])
        delaunay_triangles = get_delaunay_triangles(rect, hull1, [hi[0] for hi in hull_index])

        cap = cv2.VideoCapture(0)
        COLORS = []

        for i in count(1):
            ret, frame = cap.read()
            if not ret:
                continue

            height, width = frame.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            frame = cv2.resize(
                frame,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                points2 = detect_facial_landmarks(frame, FACE_DOWNSAMPLE_RATIO)
            except Exception as e:
                print(e)
            else:

                hull2 = [points2[hull_index_element[0]] for hull_index_element in hull_index]
                original_hull2 = [
                    points2[hull_index_element[0]] for hull_index_element in original_hull_index
                ]

                img1_warped = np.float32(frame)

                for triangle in delaunay_triangles:
                    mouth_points_set = set(mp[0] for mp in mouth_points)
                    if (
                            triangle[0] in mouth_points_set
                            and triangle[1] in mouth_points_set
                            and triangle[2] in mouth_points_set
                    ):
                        continue

                    t1 = [points1[triangle[0]], points1[triangle[1]], points1[triangle[2]]]
                    t2 = [points2[triangle[0]], points2[triangle[1]], points2[triangle[2]]]

                    warp_triangle(img1, img1_warped, t1, t2)

                mask = np.zeros(frame.shape, dtype=frame.dtype)
                cv2.fillConvexPoly(mask, np.int32(original_hull2), (255, 255, 255))
                bb = cv2.boundingRect(np.float32([original_hull2]))

                center = (bb[0] + int(bb[2] / 2), bb[1] + int(bb[3] / 2))
                frame = cv2.seamlessClone(
                    np.uint8(img1_warped), frame, mask, center, cv2.NORMAL_CLONE
                )
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/optical_flow_and_seamless_cloning")
def optical_flow_and_seamless_cloning():
    return render_template("optical_flow_and_seamless_cloning.html")


@app.route("/optical_flow_and_seamless_cloning_feed", methods=["GET", "POST"])
def optical_flow_and_seamless_cloning_feed():
    FACE_DOWNSAMPLE_RATIO = 1.5

    def detect_facial_landmarks(img, FACE_DOWNSAMPLE_RATIO=1):
        small_img = cv2.resize(
            img,
            None,
            fx=1.0 / FACE_DOWNSAMPLE_RATIO,
            fy=1.0 / FACE_DOWNSAMPLE_RATIO,
            interpolation=cv2.INTER_LINEAR,
        )

        # use the biggest face
        rect = max(detector(small_img), key=lambda r: r.area())

        scaled_rect = dlib.rectangle(
            int(rect.left() * FACE_DOWNSAMPLE_RATIO),
            int(rect.top() * FACE_DOWNSAMPLE_RATIO),
            int(rect.right() * FACE_DOWNSAMPLE_RATIO),
            int(rect.bottom() * FACE_DOWNSAMPLE_RATIO),
        )
        landmarks = predictor(img, scaled_rect)

        return [(point.x, point.y) for point in landmarks.parts()]

    def get_delaunay_triangles(rect, points, indexes):
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(points)

        found_triangles = subdiv.getTriangleList()

        delaunay_triangles = []

        def contains(rect, point):
            return (
                    rect[0] < point[0] < rect[0] + rect[2]
                    and rect[1] < point[1] < rect[1] + rect[3]
            )

        for t in found_triangles:
            triangle = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]

            # `getTriangleList` return triangles only, without origin points indices and we need them
            # so they correspond to other picture through index. So we're looking for original
            # index number for every point.
            if (
                    contains(rect, triangle[0])
                    and contains(rect, triangle[1])
                    and contains(rect, triangle[2])
            ):

                indices = []
                for index, point in enumerate(points):
                    if (
                            triangle[0][0] == point[0]
                            and triangle[0][1] == point[1]
                            or triangle[1][0] == point[0]
                            and triangle[1][1] == point[1]
                            or triangle[2][0] == point[0]
                            and triangle[2][1] == point[1]
                    ):
                        indices.append(indexes[index])

                    if len(indices) == 3:
                        delaunay_triangles.append(indices)
                        continue

        # remove duplicates
        return list(set(tuple(t) for t in delaunay_triangles))

    def warp_triangle(img1, img2, t1, t2):
        # https://www.learnopencv.com/warp-one-triangle-to-another-using-opencv-c-python/
        bb1 = cv2.boundingRect(np.float32([t1]))

        img1_cropped = img1[bb1[1]: bb1[1] + bb1[3], bb1[0]: bb1[0] + bb1[2]]

        bb2 = cv2.boundingRect(np.float32([t2]))

        t1_offset = [
            ((t1[0][0] - bb1[0]), (t1[0][1] - bb1[1])),
            ((t1[1][0] - bb1[0]), (t1[1][1] - bb1[1])),
            ((t1[2][0] - bb1[0]), (t1[2][1] - bb1[1])),
        ]
        t2_offset = [
            ((t2[0][0] - bb2[0]), (t2[0][1] - bb2[1])),
            ((t2[1][0] - bb2[0]), (t2[1][1] - bb2[1])),
            ((t2[2][0] - bb2[0]), (t2[2][1] - bb2[1])),
        ]
        mask = np.zeros((bb2[3], bb2[2], 3), dtype=np.float32)

        cv2.fillConvexPoly(mask, np.int32(t2_offset), (1.0, 1.0, 1.0), cv2.LINE_AA)

        size = (bb2[2], bb2[3])

        mat = cv2.getAffineTransform(np.float32(t1_offset), np.float32(t2_offset))

        img2_cropped = cv2.warpAffine(
            img1_cropped,
            mat,
            (size[0], size[1]),
            None,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        img2_cropped = img2_cropped * mask

        img2_cropped_slice = np.index_exp[
                             bb2[1]: bb2[1] + bb2[3], bb2[0]: bb2[0] + bb2[2]
                             ]
        img2[img2_cropped_slice] = img2[img2_cropped_slice] * ((1.0, 1.0, 1.0) - mask)
        img2[img2_cropped_slice] = img2[img2_cropped_slice] + img2_cropped

    def gen_frames():
        img1 = cv2.imread("image6.jpeg")
        height, width = img1.shape[:2]
        IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
        img1 = cv2.resize(
            img1,
            None,
            fx=1.0 / IMAGE_RESIZE,
            fy=1.0 / IMAGE_RESIZE,
            interpolation=cv2.INTER_LINEAR,
        )
        points1 = detect_facial_landmarks(img1, FACE_DOWNSAMPLE_RATIO)

        original_hull_index = cv2.convexHull(np.array(points1), returnPoints=False)
        mouth_points = [
            # [48],  # <outer mouth>
            # [49],
            # [50],
            # [51],
            # [52],
            # [53],
            # [54],
            # [55],
            # [56],
            # [57],
            # [58],  # </outer mouth>
            [60],  # <inner mouth>
            [61],
            [62],
            [63],
            [64],
            [65],
            [66],
            [67],  # </inner mouth>
        ]

        hull1 = []
        hull_index = np.concatenate((original_hull_index, mouth_points))

        # Helper map to find proper list indexes for hull2 using landmarks number
        landmark_idx_to_list_idx = {elem[0]: i for i, elem in enumerate(hull_index)}

        hull1 = [points1[hull_index_element[0]] for hull_index_element in hull_index]

        rect = (0, 0, img1.shape[1], img1.shape[0])
        delaunay_triangles = get_delaunay_triangles(rect, hull1, [hi[0] for hi in hull_index])

        cap = cv2.VideoCapture(0)
        sigma = 100
        first_frame = False

        for i in count(1):
            ret, img2 = cap.read()
            if not ret:
                continue

            height, width = img2.shape[:2]
            IMAGE_RESIZE = np.float32(height) / RESIZE_HEIGHT
            img2 = cv2.resize(
                img2,
                None,
                fx=1.0 / IMAGE_RESIZE,
                fy=1.0 / IMAGE_RESIZE,
                interpolation=cv2.INTER_LINEAR,
            )
            try:
                points2 = detect_facial_landmarks(img2, FACE_DOWNSAMPLE_RATIO)
            except Exception:
                pass
            else:

                hull2 = [points2[hull_index_element[0]] for hull_index_element in hull_index]
                original_hull2 = [
                    points2[hull_index_element[0]] for hull_index_element in original_hull_index
                ]

                img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

                if first_frame is False:
                    hull2_prev = np.array(hull2, np.float32)
                    img2_gray_prev = np.copy(img2_gray)
                    first_frame = True

                hull2_next, *_ = cv2.calcOpticalFlowPyrLK(
                    img2_gray_prev,
                    img2_gray,
                    hull2_prev,
                    np.array(hull2, np.float32),
                    winSize=(101, 101),
                    maxLevel=5,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.001),
                )

                for i, _ in enumerate(hull2):
                    hull2[i] = 0.3 * np.array(hull2[i]) + 0.7 * hull2_next[i]

                hull2_prev = np.array(hull2, np.float32)
                img2_gray_prev = img2_gray

                img1_warped = np.copy(img2)
                img1_warped = np.float32(img1_warped)

                for triangle in delaunay_triangles:
                    mouth_points_set = set(mp[0] for mp in mouth_points)
                    if (
                            triangle[0] in mouth_points_set
                            and triangle[1] in mouth_points_set
                            and triangle[2] in mouth_points_set
                    ):
                        continue
                    t1 = [points1[triangle[0]], points1[triangle[1]], points1[triangle[2]]]
                    t2 = [
                        hull2[landmark_idx_to_list_idx[triangle[0]]],
                        hull2[landmark_idx_to_list_idx[triangle[1]]],
                        hull2[landmark_idx_to_list_idx[triangle[2]]],
                    ]

                    warp_triangle(img1, img1_warped, t1, t2)

                mask = np.zeros(img2.shape, dtype=img2.dtype)
                cv2.fillConvexPoly(mask, np.int32(original_hull2), (255, 255, 255))
                r = cv2.boundingRect(np.float32([original_hull2]))

                center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
                img2 = cv2.seamlessClone(
                    np.uint8(img1_warped), img2, mask, center, cv2.NORMAL_CLONE
                )
            ret, buffer = cv2.imencode('.jpg', img2)
            img2 = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img2 + b'\r\n')

    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/controls', methods=['POST'])
def controls():
    # set the initial video feed to the default camera
    cap = cv2.VideoCapture(0)

    selected_option = request.form['options']

    if selected_option == "detect_face":
        # Process for detect_face
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("detect_face"))

    elif selected_option == "face_alignment":
        # Process for face_alignment
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("face_alignment"))

    elif selected_option == 'detect_face_rectangle':
        # Process for detect_face_rectangle
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("index"))

    elif selected_option == "find_convex_hull":
        # Process for find_convex_hull_feed
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("find_convex_hull"))

    elif selected_option == "affine_wrap_triangles":
        # Process for find_convex_hull_feed
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("affine_wrap_triangles"))

    elif selected_option == "delaunay_triangulation":
        # Process for delaunay_triangulation
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("delaunay_triangulation"))

    elif selected_option == "seamless_cloning":
        # Process for seamless_cloning
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("seamless_cloning"))

    elif selected_option == "optical_flow_and_seamless_cloning":
        # Process for optical_flow_and_seamless_cloning
        # release OpenCV resources
        cap.release()
        cv2.destroyAllWindows()
        return redirect(url_for("optical_flow_and_seamless_cloning"))


if __name__ == '__main__':
    app.run(debug=True)
