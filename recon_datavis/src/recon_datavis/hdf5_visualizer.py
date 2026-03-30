import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import os

from recon_datavis.gps.plotter import GPSPlotter
from recon_datavis import pyblit
from recon_datavis.utils import Getch, bytes2im


class HDF5Visualizer(object):

    def __init__(self, hdf5_fnames):
        self._hdf5_fnames = hdf5_fnames

        self._curr_hdf5_idx = 0
        self._prev_hdf5_idx = -1
        self._curr_hdf5_timestep = 0
        self._curr_hdf5 = None
        self._curr_hdf5_len = None
        self._load_hdf5()

        self._setup_visualization()

    def _setup_visualization(self):
        self._f, axes = plt.subplots(2, 4, figsize=(20, 10))
        ((ax_rgb_left, ax_rgb_right, ax_thermal, ax_lidar),
         (self._ax_gpscompass, ax_coll, ax_imu, ax_speedsteer)) = axes
        self._plot_is_showing = False

        self._pyblit_rgb_left = pyblit.Imshow(ax_rgb_left)
        self._pyblit_rgb_left_ax = pyblit.Axis(ax_rgb_left, [self._pyblit_rgb_left])

        self._pyblit_rgb_right = pyblit.Imshow(ax_rgb_right)
        self._pyblit_rgb_right_illuminance = pyblit.Text(ax_rgb_right)
        self._pyblit_rgb_right_ax = pyblit.Axis(ax_rgb_right, [self._pyblit_rgb_right, self._pyblit_rgb_right_illuminance])

        self._pyblit_thermal = pyblit.Imshow(ax_thermal)
        self._pyblit_thermal_text = pyblit.Text(ax_thermal)
        self._pyblit_thermal_ax = pyblit.Axis(ax_thermal, [self._pyblit_thermal, self._pyblit_thermal_text])

        self._pyblit_lidar = pyblit.Scatter(ax_lidar)
        self._pyblit_lidar_ax = pyblit.Axis(ax_lidar, [self._pyblit_lidar])

        self._gps_plotter = GPSPlotter()

        self._pyblit_coll = pyblit.Bar(ax_coll)
        self._pyblit_coll_ax = pyblit.Axis(ax_coll, [self._pyblit_coll])

        self._pyblit_imu = pyblit.Bar(ax_imu)
        self._pyblit_imu_jackal = pyblit.Bar(ax_imu)
        self._pyblit_imu_ax = pyblit.Axis(ax_imu, [self._pyblit_imu, self._pyblit_imu_jackal])

        self._pyblit_speedsteer_commanded_linvel = pyblit.Line(ax_speedsteer)
        self._pyblit_speedsteer_commanded_angvel = pyblit.Line(ax_speedsteer)
        self._pyblit_speedsteer_actual_linvel = pyblit.Line(ax_speedsteer)
        self._pyblit_speedsteer_actual_angvel = pyblit.Line(ax_speedsteer)
        self._pyblit_speedsteer_legend = pyblit.Legend(ax_speedsteer)
        self._pyblit_speedsteer_ax = pyblit.Axis(ax_speedsteer,
                                                 [self._pyblit_speedsteer_commanded_linvel,
                                                  self._pyblit_speedsteer_commanded_angvel,
                                                  self._pyblit_speedsteer_actual_linvel,
                                                  self._pyblit_speedsteer_actual_angvel,
                                                  self._pyblit_speedsteer_legend])

    ############
    ### Hdf5 ###
    ############

    def _load_hdf5(self):
        self._curr_hdf5 = h5py.File(self._hdf5_fnames[self._curr_hdf5_idx], 'r')
        self._curr_hdf5_len = len(self._curr_hdf5['collision/any'])

    def _next_timestep(self):
        if (self._curr_hdf5_timestep == self._curr_hdf5_len - 1) and (self._curr_hdf5_idx == len(self._hdf5_fnames) - 1):
            return # at the end, do nothing

        self._curr_hdf5_timestep += 1
        if self._curr_hdf5_timestep >= self._curr_hdf5_len:
            self._curr_hdf5_idx += 1
            self._load_hdf5()
            self._curr_hdf5_timestep = 0

    def _prev_timestep(self):
        if (self._curr_hdf5_timestep == 0) and (self._curr_hdf5_idx == 0):
            return # at the beginning, do nothing

        self._curr_hdf5_timestep -= 1
        if self._curr_hdf5_timestep < 0:
            self._curr_hdf5_idx -= 1
            self._load_hdf5()
            self._curr_hdf5_timestep = self._curr_hdf5_len - 1

    def _next_hdf5(self):
        if self._curr_hdf5_idx == len(self._hdf5_fnames) - 1:
            return # at the end, do nothing

        self._curr_hdf5_idx += 1
        self._load_hdf5()
        self._curr_hdf5_timestep = 0

    def _prev_hdf5(self):
        if self._curr_hdf5_idx == 0:
            return # at the beginning, do nothing

        self._curr_hdf5_idx -= 1
        self._load_hdf5()
        self._curr_hdf5_timestep = 0

    def _next_hdf5_end(self):
        if self._curr_hdf5_idx == len(self._hdf5_fnames) - 1:
            pass # at the end, do nothing
        else:
            self._curr_hdf5_idx += 1

        self._load_hdf5()
        self._curr_hdf5_timestep = self._curr_hdf5_len - 1

    def _prev_hdf5_end(self):
        if self._curr_hdf5_idx == 0:
            pass  # at the end, do nothing
        else:
            self._curr_hdf5_idx -= 1

        self._load_hdf5()
        self._curr_hdf5_timestep = self._curr_hdf5_len - 1

    #################
    ### Visualize ###
    #################

    def _get_hdf5_topic(self, topic):
        value = self._curr_hdf5[topic][self._curr_hdf5_timestep]
        if type(value) == np.string_:
            value = bytes2im(value)
        return value

    def _plot_lidar(self):
        lidar = self._get_hdf5_topic('lidar')
        assert len(lidar) == 360

        measurement_indices = list(range(300, 360)) + list(range(0, 60))
        angles = np.deg2rad(np.linspace(30, 150., 120))
        lidar = lidar[measurement_indices]
        lidar_notfinite = np.logical_not(np.isfinite(lidar))
        max_range = 15.
        lidar[lidar_notfinite] = max_range

        x = lidar * np.cos(angles)
        y = lidar * np.sin(angles)
        c = [to_rgb('r') if notfinite else to_rgb('k') for notfinite in lidar_notfinite]

        self._pyblit_lidar.draw(x, y, c=c, s=2.)
        self._pyblit_lidar_ax.ax.set_xlim((-16, 16))
        self._pyblit_lidar_ax.ax.set_ylim((0, 16))
        self._pyblit_lidar_ax.ax.set_aspect('equal')
        self._pyblit_lidar_ax.draw()

    def _plot_speedsteer(self):
        commanded_linvel = self._get_hdf5_topic('commands/linear_velocity')
        commanded_angvel = -self._get_hdf5_topic('commands/angular_velocity')
        actual_linvel = self._get_hdf5_topic('jackal/linear_velocity')
        actual_angvel = -self._get_hdf5_topic('jackal/angular_velocity')

        self._pyblit_speedsteer_commanded_linvel.draw([0, 0], [0, commanded_linvel],
                                                      linestyle='-', color='k', linewidth=10.0, label='Commanded')
        self._pyblit_speedsteer_commanded_angvel.draw([0, commanded_angvel], [0, 0],
                                                      linestyle='-', color='k', linewidth=10.0)
        self._pyblit_speedsteer_actual_linvel.draw([0, 0], [0, actual_linvel],
                                                   linestyle='-', color='c', linewidth=4.0, label='Actual')
        self._pyblit_speedsteer_actual_angvel.draw([0, actual_angvel], [0, 0],
                                                   linestyle='-', color='c', linewidth=4.0)
        self._pyblit_speedsteer_legend.draw(loc='lower left')
        ax = self._pyblit_speedsteer_ax.ax
        ax.set_xlim((-1.5, 1.5))
        ax.set_ylim((-1.5, 1.5))
        ax.set_title('Speed / Steer')
        self._pyblit_speedsteer_ax.draw()

    def _plot_collision(self):
        topics = ['collision/{0}'.format(name) for name in
                  ('any', 'physical', 'close', 'flipped', 'stuck', 'outside_geofence')]
        colls = [self._get_hdf5_topic(topic) for topic in topics]

        self._pyblit_coll.draw(
            np.arange(len(colls)),
            colls,
            tick_label=[topic.replace('collision/', '').replace('outside_', '') for topic in topics],
            color='r'
        )
        ax = self._pyblit_coll_ax.ax
        ax.set_title('Collision')
        ax.set_xlim((-0.5, len(colls) + 0.5))
        ax.set_ylim((0, 1))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        self._pyblit_coll_ax.draw()

    def _plot_imu(self):
        linacc = self._get_hdf5_topic('imu/linear_acceleration')
        angvel = self._get_hdf5_topic('imu/angular_velocity')
        jackal_linacc = self._get_hdf5_topic('jackal/imu/linear_acceleration')
        jackal_angvel = self._get_hdf5_topic('jackal/imu/angular_velocity')

        linacc[2] = -linacc[2] + 9.81 # negative since upside down, and subtract off gravity offset
        jackal_linacc[2] -= 9.81

        self._pyblit_imu.draw(
            np.arange(6),
            np.concatenate((linacc, angvel)),
            tick_label=['linacc', '', '', 'angvel', '', ''],
            color='k',
            width=0.8
        )
        self._pyblit_imu_jackal.draw(
            np.arange(6),
            np.concatenate((jackal_linacc, jackal_angvel)),
            tick_label=['linacc', '', '', 'angvel', '', ''],
            color='r',
            width=0.5
        )
        ax = self._pyblit_imu_ax.ax
        ax.set_title('IMU')
        ax.set_xlim((-0.5, 6.5))
        ax.set_ylim((-3., 3.))
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        self._pyblit_imu_ax.draw()

    def _plot_gpscompass(self):
        compass_bearing = self._get_hdf5_topic('imu/compass_bearing')
        latlong = self._get_hdf5_topic('gps/latlong')
        if not np.isfinite(latlong).all():
            latlong = self._gps_plotter.SE_LATLONG
            color = (1., 1., 1., 0.)
        else:
            color = 'r'
        self._gps_plotter.plot_latlong_and_compass_bearing(self._ax_gpscompass, latlong, compass_bearing, color=color)

    def _update_visualization(self):
        self._pyblit_rgb_left.draw(self._get_hdf5_topic('images/rgb_left'))
        self._pyblit_rgb_left_ax.draw()

        self._pyblit_rgb_right.draw(self._get_hdf5_topic('images/rgb_right'))
        self._pyblit_rgb_right_illuminance.draw(
            0.25,
            0.0,
            text='Illuminance: {0:.0f}'.format(self._get_hdf5_topic('android/illuminance')),
            transform=self._pyblit_rgb_right_ax.ax.transAxes, fontsize=11, color='w')
        self._pyblit_rgb_right_ax.draw()

        self._pyblit_thermal.draw(self._get_hdf5_topic('images/thermal'))
        self._pyblit_thermal_text.draw(
            0.05,
            0.9,
            text='{4}\n{0}/{1}, {2}/{3}'.format(self._curr_hdf5_timestep + 1, self._curr_hdf5_len,
                                                          self._curr_hdf5_idx + 1, len(self._hdf5_fnames),
                                                          os.path.basename(self._hdf5_fnames[self._curr_hdf5_idx])),
            transform=self._pyblit_thermal_ax.ax.transAxes, fontsize=11, color='w')
        self._pyblit_thermal_ax.draw()

        self._plot_lidar()


        self._plot_speedsteer()
        self._plot_collision()
        self._plot_imu()
        self._plot_gpscompass()

        if not self._plot_is_showing:
            plt.show(block=False)
            self._plot_is_showing = True
            plt.pause(0.01)
        self._f.canvas.flush_events()

    ###########
    ### Run ###
    ###########

    def run(self):
        self._update_visualization()

        while True:
            print('{0}/{1}, {2}/{3}'.format(self._curr_hdf5_timestep+1, self._curr_hdf5_len,
                                            self._curr_hdf5_idx+1, len(self._hdf5_fnames)))
            self._update_visualization()

            char = Getch.getch()
            self._prev_hdf5_idx = self._curr_hdf5_idx

            if char == 'q':
                break
            elif char == 'e':
                self._next_timestep()
            elif char == 'w':
                self._prev_timestep()
            elif char == 'd':
                self._next_hdf5()
            elif char == 's':
                self._prev_hdf5()
            elif char >= '1' and char <= '9':
                for _ in range(int(char)):
                    self._next_timestep()
            elif char == 'c':
                self._next_hdf5_end()
            elif char == 'x':
                self._prev_hdf5_end()
            else:
                continue
