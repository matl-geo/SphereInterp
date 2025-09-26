import numpy as np
from sklearn.neighbors import BallTree

def lat_lon_to_radians(lat_lon):
    return np.radians(lat_lon)

def makeBallTree(grid_radians):
    tree = BallTree(grid_radians,metric='haversine')
    return tree

def findNearestNeighbour(station_point_radians,tree,grid):
    distances, nn = tree.query(station_point_radians, k=4)
    
    # Unravel into original grid
    nn_idx = np.unravel_index(nn, grid)

    # Find lower left grid cell
    x_lower_left = np.min(nn_idx[1])
    y_lower_left = np.min(nn_idx[0])
    
    return x_lower_left, y_lower_left

def altBilinearInterpolation(x,y,station_lat,station_lon,model_lat,model_lon,model_value):

    #Define grid points around station
    p1 = (y,x)
    p2 = (y,x+1)
    p3 = (y+1,x)
    p4 = (y+1,x+1)

    # Define vectors
    x_vec = [model_lon[p1],
            model_lon[p2],
            model_lon[p3],
            model_lon[p4]]

    y_vec = [model_lat[p1],
            model_lat[p2],
            model_lat[p3],
            model_lat[p4]]

    xy_vec = [x_vec[0]*y_vec[0],
            x_vec[1]*y_vec[1],
            x_vec[2]*y_vec[2],
            x_vec[3]*y_vec[3]]

    f_vec = [model_value[p1],
            model_value[p2],
            model_value[p3],
            model_value[p4]]

    ones_vec = [1,
                1,
                1,
                1]

    #Construct matrix
    D  = np.matrix([ones_vec,x_vec,y_vec,xy_vec]).T
    D1 = np.matrix([f_vec,x_vec,y_vec,xy_vec]).T
    D2 = np.matrix([ones_vec,f_vec,y_vec,xy_vec]).T
    D3 = np.matrix([ones_vec,x_vec,f_vec,xy_vec]).T
    D4 = np.matrix([ones_vec,x_vec,y_vec,f_vec]).T

    #Compute determinant of the matrix
    D_det  = np.linalg.det(D)
    D1_det = np.linalg.det(D1)
    D2_det = np.linalg.det(D2)
    D3_det = np.linalg.det(D3)
    D4_det = np.linalg.det(D4)

    #Compute coefficients
    a = D1_det/D_det
    b = D2_det/D_det
    c = D3_det/D_det
    d = D4_det/D_det

    #Station point alternative biliniar interpolation
    interpolated_value = a + b*station_lon + c*station_lat + d*station_lon*station_lat

    return interpolated_value


def runAltBilinearInterpolation(station_lat,station_lon,model_lat,model_lon,model_value):
    
    #Convert model grid to radians
    grid_points = np.vstack([model_lat.ravel(),model_lon.ravel()]).T
    grid_points_radians = lat_lon_to_radians(grid_points)

    #Make BallTree
    tree = makeBallTree(grid_points_radians)

    #Initialize arrays
    n_stations = len(station_lat)
    interpolated_value = np.zeros(n_stations)
    x_list = np.zeros(n_stations)
    y_list = np.zeros(n_stations)

    #Interpolation loop
    for i in range(0,n_stations):

        #Convert station point to radians
        station_point = [[station_lat[i],station_lon[i]]]
        station_point_radians = lat_lon_to_radians(station_point)

        #Find nearest grid point
        x,y = findNearestNeighbour(station_point_radians,tree,np.shape(model_lat))

        #Alternative bilinear interpolation
        interpolated_value[i] = altBilinearInterpolation(x,y,station_lat[i],station_lon[i],model_lat,model_lon,model_value)

        #Save x and y
        x_list[i] = x
        y_list[i] = y

    return interpolated_value, x_list, y_list