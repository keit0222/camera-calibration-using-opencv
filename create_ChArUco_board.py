import cv2

aruco = cv2.aruco

pixel_per_onebox = 256

dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_100)
board = aruco.CharucoBoard_create(5, 4, 0.06, 0.04, dictionary) # squaresX, squaresY, squareLength, markerLength, dictionary
boardImage=0
boardImage = board.draw((pixel_per_onebox*5,pixel_per_onebox*8), boardImage, 0, 1)
cv2.imwrite("charuco2.bmp", boardImage)
cv2.imshow("charuco", boardImage)
cv2.waitKey(0)
cv2.destroyAllWindows()