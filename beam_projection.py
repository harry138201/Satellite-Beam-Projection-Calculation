import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

R_EARTH = 6371.0
H_GEO = 42164.0


def cartesian_to_geo(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    lat = np.arcsin(z / r)
    lon = np.arctan2(y, x)
    return np.degrees(lat), np.degrees(lon)


def geo_to_cartesian(lat, lon, r=R_EARTH):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = r * np.cos(lat_rad) * np.cos(lon_rad)
    y = r * np.cos(lat_rad) * np.sin(lon_rad)
    z = r * np.sin(lat_rad)
    return x, y, z


def get_rotation_matrix(lat0, lon0):
    # Direct construction: R = [north, east, target]
    # After rotation:
    # - local x-axis -> global north direction at subpoint
    # - local y-axis -> global east direction at subpoint
    # - local z-axis -> global direction from Earth center to subpoint
    cl = np.cos(lat0)
    sl = np.sin(lat0)
    cln = np.cos(lon0)
    sln = np.sin(lon0)
    
    # north direction vector (unit)
    north = np.array([
        -sl * cln,
        -sl * sln,
        cl
    ])
    
    # east direction vector (unit)
    east = np.array([
        -sln,
        cln,
        0.0
    ])
    
    # target direction (Earth center to subpoint, unit)
    target = np.array([
        cl * cln,
        cl * sln,
        sl
    ])
    
    # Construct rotation matrix - each column is the rotated basis vector
    R = np.column_stack([north, east, target])
    
    return R


def rotate_point(p, lat0, lon0):
    R = get_rotation_matrix(lat0, lon0)
    return R @ p


def compute_beam_projection_local(
    R=R_EARTH,
    H=H_GEO,
    theta_deg=1.0,
    beta_deg=5.0,
    alpha_deg=0.0,
    num_points=72
):
    theta = np.radians(theta_deg)
    beta = np.radians(beta_deg)
    alpha = np.radians(alpha_deg)

    a = np.array([
        np.sin(beta) * np.cos(alpha),
        np.sin(beta) * np.sin(alpha),
        -np.cos(beta)
    ])
    a = a / np.linalg.norm(a)
    S = np.array([0, 0, H])

    if np.abs(beta) < 1e-6:
        v1 = np.array([1, 0, 0])
    else:
        temp = np.array([0, 0, 1]) - np.dot(a, np.array([0, 0, 1])) * a
        v1 = temp / np.linalg.norm(temp)
    v2 = np.cross(a, v1)

    points = []
    for i in range(num_points):
        phi = 2 * np.pi * i / num_points
        u = v1 * np.cos(phi) + v2 * np.sin(phi)
        dir_vec = a * np.cos(theta) + u * np.sin(theta)
        dir_vec = dir_vec / np.linalg.norm(dir_vec)

        a_coeff = 1.0
        b_coeff = 2 * np.dot(S, dir_vec)
        c_coeff = np.linalg.norm(S)**2 - R**2
        delta = b_coeff**2 - 4 * a_coeff * c_coeff

        if delta < 0:
            continue

        t1 = (-b_coeff - np.sqrt(delta)) / (2 * a_coeff)
        t2 = (-b_coeff + np.sqrt(delta)) / (2 * a_coeff)

        valid_ts = [t for t in [t1, t2] if t > 0]
        if valid_ts:
            t = min(valid_ts)
            P = S + t * dir_vec
            points.append(P)

    return np.array(points)


def calculate_beam_offset_from_center(
    satellite_height,
    subpoint_lat,
    subpoint_lon,
    beam_center_lat,
    beam_center_lon,
    earth_radius=R_EARTH
):
    # 卫星位置（地心坐标系：星下点正上方）
    sat_x, sat_y, sat_z = geo_to_cartesian(subpoint_lat, subpoint_lon, satellite_height)
    S = np.array([sat_x, sat_y, sat_z])
    
    # 星下点和波束中心点的地心坐标
    sub_x, sub_y, sub_z = geo_to_cartesian(subpoint_lat, subpoint_lon, earth_radius)
    c_x, c_y, c_z = geo_to_cartesian(beam_center_lat, beam_center_lon, earth_radius)
    
    # 卫星到两个点的向量（卫星指向地面）
    vec_S_sub = np.array([sub_x, sub_y, sub_z]) - S
    vec_S_c = np.array([c_x, c_y, c_z]) - S
    
    # 单位化
    vec_S_sub_unit = vec_S_sub / np.linalg.norm(vec_S_sub)
    vec_S_c_unit = vec_S_c / np.linalg.norm(vec_S_c)
    
    # 计算极角β：两个向量的夹角
    cos_beta = np.dot(vec_S_sub_unit, vec_S_c_unit)
    beta = np.arccos(np.clip(cos_beta, -1, 1))
    beta_deg = np.degrees(beta)
    
    # 极角为0时直接返回
    if abs(beta_deg) < 1e-6:
        return 0.0, 0.0
    
    # ============== 计算方位角α ==============
    # 地理正北方向单位向量（地心坐标系）
    north_x, north_y, north_z = geo_to_cartesian(subpoint_lat + 0.1, subpoint_lon, earth_radius)
    vec_north = np.array([north_x, north_y, north_z]) - np.array([sub_x, sub_y, sub_z])
    vec_north = vec_north / np.linalg.norm(vec_north)
    
    # 地理正东方向单位向量
    east_x, east_y, east_z = geo_to_cartesian(subpoint_lat, subpoint_lon + 0.1 / np.cos(np.radians(subpoint_lat)), earth_radius)
    vec_east = np.array([east_x, east_y, east_z]) - np.array([sub_x, sub_y, sub_z])
    vec_east = vec_east / np.linalg.norm(vec_east)
    
    # 波束中心相对于星下点的向量（地面局部坐标系）
    vec_offset = np.array([c_x, c_y, c_z]) - np.array([sub_x, sub_y, sub_z])
    vec_offset = vec_offset - np.dot(vec_offset, vec_S_sub_unit) * vec_S_sub_unit  # 投影到切平面
    vec_offset_unit = vec_offset / np.linalg.norm(vec_offset) if np.linalg.norm(vec_offset) > 1e-6 else vec_north
    
    # 计算方位角：正北为0，顺时针增加
    proj_north = np.dot(vec_offset_unit, vec_north)
    proj_east = np.dot(vec_offset_unit, vec_east)
    alpha = np.arctan2(proj_east, proj_north)
    alpha_deg = np.degrees(alpha)
    
    # 调整范围到0~360°
    if alpha_deg < 0:
        alpha_deg += 360
    
    return beta_deg, alpha_deg


def calculate_beam_boundary(
    satellite_height,
    subpoint_lat,
    subpoint_lon,
    beam_half_angle,
    beam_offset_beta=None,
    beam_offset_alpha=None,
    beam_center_lat=None,
    beam_center_lon=None,
    num_points=72,
    earth_radius=R_EARTH
):
    if beam_center_lat is not None and beam_center_lon is not None:
        beta_deg, alpha_deg = calculate_beam_offset_from_center(
            satellite_height,
            subpoint_lat, subpoint_lon,
            beam_center_lat, beam_center_lon,
            earth_radius
        )
    else:
        if beam_offset_beta is None:
            beta_deg = 0.0
        else:
            beta_deg = beam_offset_beta
        if beam_offset_alpha is None:
            alpha_deg = 0.0
        else:
            alpha_deg = beam_offset_alpha

    # 转换方位角：地理方位角（正北为0，顺时针）转本地坐标系方位角
    # In local coordinates after rotation:
    # - original x-axis points north, original y-axis points east (right hand: x north, y east)
    # - Geographic alpha: clockwise from north (0° = north, 90° = east)
    # - Mathematical alpha in polar coordinates: counter-clockwise from x-axis
    # With x north, y east: 0° north (geo) = 0° (math), 90° east (geo) = 90° (math)
    # They are equal because y east is clockwise from x north in this coordinate system
    compute_alpha_deg = alpha_deg
    points_cart = compute_beam_projection_local(
        R=earth_radius,
        H=satellite_height,
        theta_deg=beam_half_angle,
        beta_deg=beta_deg,
        alpha_deg=compute_alpha_deg,
        num_points=num_points
    )

    lat0_rad = np.radians(subpoint_lat)
    lon0_rad = np.radians(subpoint_lon)

    boundary = []
    R = get_rotation_matrix(lat0_rad, lon0_rad)
    for p in points_cart:
        # compute_beam_projection_local gives points in local coordinates where:
        # - origin is at Earth center
        # - local z-axis points directly at target subpoint (from origin)
        # - R is constructed to rotate the original (global) z-axis (0,0,1) to target direction
        # - Therefore, p_global = R @ p_local rotates from local to global coordinates
        p_global = R @ p
        lat, lon = cartesian_to_geo(p_global[0], p_global[1], p_global[2])
        boundary.append((lat, lon))

    return np.array(boundary)


def plot_projection(points_cart, R=R_EARTH, H=H_GEO):
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title('3D View: Satellite and Beam Projection')
    ax1.set_xlabel('X (km)')
    ax1.set_ylabel('Y (km)')
    ax1.set_zlabel('Z (km)')
    ax1.set_box_aspect([1, 1, 1])

    u_sphere = np.linspace(0, 2 * np.pi, 64)
    v_sphere = np.linspace(0, np.pi, 32)
    x_sphere = R * np.outer(np.cos(u_sphere), np.sin(v_sphere))
    y_sphere = R * np.outer(np.sin(u_sphere), np.sin(v_sphere))
    z_sphere = R * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='skyblue', alpha=0.3)

    if len(points_cart) > 0:
        ax1.plot(points_cart[:, 0], points_cart[:, 1], points_cart[:, 2], 'ro-', markersize=4, label='Projection Boundary')
        ax1.plot(points_cart[[0, -1], 0], points_cart[[0, -1], 1], points_cart[[0, -1], 2], 'ro-')

    ax1.scatter([0], [0], [H], color='blue', s=100, label='Satellite')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.set_title('Beam Projection on Earth (2D Lat/Lon)')
    ax2.set_xlabel('Longitude (deg)')
    ax2.set_ylabel('Latitude (deg)')
    ax2.grid(True)

    if len(points_cart) > 0:
        lats, lons = [], []
        for p in points_cart:
            lat, lon = cartesian_to_geo(p[0], p[1], p[2])
            lats.append(lat)
            lons.append(lon)
        lats.append(lats[0])
        lons.append(lons[0])
        ax2.plot(lons, lats, 'bo-', markersize=4, label='Projection Boundary')
        ax2.legend()
        ax2.set_xlim(-180, 180)
        ax2.set_ylim(-90, 90)

    plt.tight_layout()
    plt.show()


def calculate_beam_area(boundary, earth_radius=R_EARTH):
    if len(boundary) < 3:
        return 0.0
    
    # 正确的球面多边形面积计算，使用高斯-博内定理
    # 参考：L'Huilier定理计算球面多边形面积
    lats = np.radians(boundary[:, 0])
    lons = np.radians(boundary[:, 1])
    n = len(boundary)
    
    # 将顶点转换为地心方向单位向量
    vertices = []
    for lat, lon in zip(lats, lons):
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        vertices.append(np.array([x, y, z]))
    
    # 计算每个内角
    total_excess = 0.0
    
    for i in range(n):
        v0 = vertices[(i - 1) % n]
        v1 = vertices[i]
        v2 = vertices[(i + 1) % n]
        
        # 计算相邻边的球面角（使用点积计算夹角）
        # 边向量通过叉乘得到垂直于面的向量
        e1 = np.cross(v0, v1)
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(v1, v2)
        e2 = e2 / np.linalg.norm(e2)
        
        # 计算内角
        cos_angle = -np.dot(e1, e2)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        total_excess += angle
    
    # 高斯-博内定理：球面多余 = 内角和 - (n - 2) * π
    spherical_excess = total_excess - (n - 2) * np.pi
    area = spherical_excess * earth_radius**2
    
    return abs(area)


def plot_beam_boundary_geo(boundary, subpoint_lat=None, subpoint_lon=None, beam_center_lat=None, beam_center_lon=None, title='Beam Projection Boundary'):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Longitude (deg)', fontsize=12)
    ax.set_ylabel('Latitude (deg)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    lats = boundary[:, 0].tolist()
    lons = boundary[:, 1].tolist()
    lats.append(lats[0])
    lons.append(lons[0])

    ax.plot(lons, lats, 'bo-', markersize=4, linewidth=1.5, label='Beam Boundary')
    ax.fill(lons, lats, 'lightblue', alpha=0.4)

    if subpoint_lat is not None and subpoint_lon is not None:
        ax.scatter([subpoint_lon], [subpoint_lat], color='green', s=100, zorder=5, label='Subpoint')

    if beam_center_lat is not None and beam_center_lon is not None:
        ax.scatter([beam_center_lon], [beam_center_lat], color='red', s=100, zorder=6, label='Beam Center')

    ax.legend(fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)

    plt.tight_layout()
    plt.show()


def main():
    print('=== Earth Synchronous Orbit Satellite Beam Projection (Test Mode) ===')
    print('Input parameters:')
    print(f'  Default earth radius: {R_EARTH} km')
    print(f'  Default satellite height: {H_GEO} km')
    print('')

    satellite_height = float(input(f'Enter satellite height (km, default={H_GEO}): ') or H_GEO)
    subpoint_lat = float(input('Enter subpoint latitude (deg, default=39.9): ') or 39.9)
    subpoint_lon = float(input('Enter subpoint longitude (deg, default=116.4): ') or 116.4)
    beam_half_angle = float(input('Enter beam half-angle (deg, default=2.0): ') or 2.0)

    print('')
    print('Choose how to set beam center:')
    print('  1. By offset angles (polar + azimuth)')
    print('  2. By ground coordinates (beam center lat/lon)')
    choice = input('Enter your choice (1 or 2, default=2): ') or '2'

    beam_center_lat = None
    beam_center_lon = None
    beam_offset_beta = None
    beam_offset_alpha = None

    if choice.strip() == '2':
        print('')
        print('Enter beam center ground coordinates:')
        beam_center_lat = float(input('  Beam center latitude (deg, default=39.9): ') or 39.9)
        beam_center_lon = float(input('  Beam center longitude (deg, default=116.4): ') or 116.4)
        print('')
        print('Calculating beam offset angles...')
        beta_deg, alpha_deg = calculate_beam_offset_from_center(
            satellite_height,
            subpoint_lat, subpoint_lon,
            beam_center_lat, beam_center_lon
        )
        print(f'  Computed beam offset polar angle: {beta_deg:.2f}°')
        print(f'  Computed beam offset azimuth: {alpha_deg:.2f}°')
    else:
        print('')
        beam_offset_beta = float(input('Enter beam offset polar angle (deg, default=0.0): ') or 0.0)
        beam_offset_alpha = float(input('Enter beam offset azimuth (deg, default=0.0): ') or 0.0)

    num_points = int(input('Enter number of sample points (default=72): ') or 72)

    print('')
    print('Computing beam projection...')
    boundary = calculate_beam_boundary(
        satellite_height=satellite_height,
        subpoint_lat=subpoint_lat,
        subpoint_lon=subpoint_lon,
        beam_half_angle=beam_half_angle,
        beam_offset_beta=beam_offset_beta,
        beam_offset_alpha=beam_offset_alpha,
        beam_center_lat=beam_center_lat,
        beam_center_lon=beam_center_lon,
        num_points=num_points
    )

    print(f'Computed {len(boundary)} points.')
    print('')

    if len(boundary) > 0:
        print('All boundary points:')
        for i, (lat, lon) in enumerate(boundary):
            print(f'  {i+1:3d}: lat={lat:.4f}°, lon={lon:.4f}°')
        
        print('')
        area_km2 = calculate_beam_area(boundary)
        print(f'✅ Beam coverage area: {area_km2:.2f} km²')
        print(f'                              ({area_km2/1000000:.4f} million km²)')

        print('')
        print('Plotting...')
        title = f'Beam Projection (Subpoint: {subpoint_lat:.1f}°N {subpoint_lon:.1f}°E), Area: {area_km2:.0f} km²'
        plot_beam_boundary_geo(
            boundary,
            subpoint_lat=subpoint_lat,
            subpoint_lon=subpoint_lon,
            beam_center_lat=beam_center_lat,
            beam_center_lon=beam_center_lon,
            title=title
        )
    else:
        print('⚠️  No intersection found!')
        print('   Possible reasons:')
        print('   - Beam half-angle is too small (beam doesn\'t reach Earth)')
        print('   - Beam offset is too large (beam misses Earth)')
        print('   - Satellite orbit height is too high')
        print('')
        print('   Suggestions:')
        print('   - Increase beam half-angle')
        print('   - Decrease beam offset polar angle')


if __name__ == '__main__':
    main()
