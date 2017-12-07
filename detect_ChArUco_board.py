import cv2
aruco = cv2.aruco

# チェッカーボードの生成 #
parameters = aruco.DetectorParameters_create()
dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
board = aruco.CharucoBoard_create(5, 8, 0.06, 0.04, dictionary) # squaresX, squaresY, squareLength, markerLength, dictionary

# 入力画像 #
import glob, os
image_paths = [os.path.basename(r) for r in glob.glob('calibration/*.bmp')]

for image_path in image_paths:
    path = 'calibration/' + image_path
    checkerBoardImage = cv2.imread(path)
    orgHeight, orgWidth = checkerBoardImage.shape[:2]
    size = (int(orgWidth/2), int(orgHeight/2))
    halfImg = cv2.resize(checkerBoardImage, size)
    cv2.imshow("test image", halfImg)
    cv2.waitKey(0)
    
    # ChArUco マーカーを検出 #
    markerCorners, markerIds  = [0,0]
    markerCorners, markerIds, rejectedImgPoints = aruco.detectMarkers(checkerBoardImage, dictionary)

    # 検出されたマーカーをもとに，チェッカーボードを検出して，結果を描画 #
    outputImage = checkerBoardImage.copy();
    if markerIds is None:
        break
    if markerIds.size > 0:
        charucoCorners, charucoIds = [0,0]
        charucoCorners, charucoIds = aruco.interpolateCornersCharuco(markerCorners, markerIds, checkerBoardImage, board, charucoCorners, charucoIds)
        outputImage = aruco.drawDetectedCornersCharuco(outputImage, charucoCorners, charucoIds)

    orgHeight, orgWidth = outputImage.shape[:2]
    size = (int(orgWidth/2), int(orgHeight/2))
    halfImg = cv2.resize(outputImage, size)
    cv2.imshow("test image", halfImg)
    cv2.waitKey(0)