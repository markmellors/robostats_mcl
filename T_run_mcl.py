import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import montecarlo_localization as mcl
import os, sys

def main(filename='./mcl_test_arena.mp4'):

    np.random.seed(5)
    loaded_map = mcl.occupancy_map('data/map/test_arena.dat')
    logdata = mcl.load_T_log('data/log/data_2020-02-22_19-55-08_998399.log')
    logdata_scans = logdata.query('type > 0.1').values
    #Initialize 100 particles uniformly in valid locations on the map
    laser = mcl.laser_sensor(stdv_cm=2, uniform_weight=0.2)
    x_initial = 200
    y_initial = 200
    theta_initial = 0
    start_pose = np.array([x_initial, y_initial, theta_initial])
    particle_list = [mcl.robot_particle(loaded_map, laser, log_prob_descale=2000,
                                        sigma_fwd_pct=0.8, sigma_theta_pct=0.02, initial_pose=start_pose)
                     for _ in range(1000)]

    fig, ax = plt.subplots(figsize=(16,9))
    pmap = ParticleMap(ax, loaded_map, particle_list,
                       target_particles=300, draw_max=2000, resample_period=1)
    # pass a generator in "emitter" to produce data for the update func
    ani = animation.FuncAnimation(fig, pmap.update, logdata_scans, interval=50,
                                  blit=False, repeat=False)
    print("Saving video to file: ", filename)
    ani.save(filename, dpi=100, fps=10, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close('all')



class ParticleMap(object):
    def __init__(self, ax, global_map, particle_list, target_particles=300,  draw_max=2000, resample_period=10):
        self.ax = ax
        self.draw_max = draw_max
        self.global_map = global_map
        self.particle_list = particle_list
        mcl.draw_map_state(global_map, particle_list, ax=self.ax, draw_max=self.draw_max)
        self.i = 1
        self.target_particles = target_particles
        self.resample_period = resample_period

    def update(self, message):
        if self.i % self.resample_period == 0:# Resample and plot state
            self.particle_list = mcl.mcl_update(self.particle_list, message, resample=True,
                                                target_particles=self.target_particles) # Update
            plt.cla()

            poses = np.vectorize(lambda p: (p.pose[0], p.pose[1]))(self.particle_list)
            avg_x = np.mean(poses[0])
            avg_y = np.mean(poses[1])
            average_particle = [avg_x, avg_y, message[5]]

            mcl.draw_map_state(self.global_map, self.particle_list, self.ax, draw_max=self.draw_max)
            mcl.plot_particle(average_particle, self.ax, pass_pose=True, color='r')
            #print(pd.Series([p.weight for p in self.particle_list]).describe())
        else: # Just update particle weights / locations - do not resample
            self.particle_list = mcl.mcl_update(self.particle_list, message,
                                                target_particles=self.target_particles) # Update
        self.i += 1
        print(self.i, end='\r')
        sys.stdout.flush()

if __name__ == "__main__":
    main()

