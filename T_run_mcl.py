import argparse
import readline
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
import montecarlo_localization as mcl
from progress.bar import Bar

def main(video_file=None, map_file=None, log_file=None, range_file=None):

    if None in [video_file, map_file, log_file, range_file]:
        raise Exception("supply all parameters to produce a video file")

    np.random.seed(5)
    loaded_map = mcl.occupancy_map(map_file, range_filename=range_file)
    logdata = mcl.load_T_log(log_file)
    logdata_scans = logdata.query('type > 0.1').values
    #Initialize 100 particles uniformly in valid locations on the map
    laser = mcl.laser_sensor(stdv_cm=10, uniform_weight=0.2)
    particle_list = [mcl.robot_particle(loaded_map, laser, log_prob_descale=2000,
                                        sigma_fwd_pct=1.5, sigma_theta_pct=0.03)
                     for _ in range(1000)]

    fig, ax = plt.subplots(figsize=(16,9))

    print("Saving video to file: ", video_file)

    pmap = ParticleMap(ax, loaded_map, particle_list, logdata_scans,
                       target_particles=300, draw_max=2000, resample_period=3)


    # pass a generator in "emitter" to produce data for the update func
    ani = animation.FuncAnimation(fig, pmap.update, logdata_scans, interval=50,
                                  blit=False, repeat=False)

    ani.save(video_file, dpi=100, fps=10, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
    plt.close('all')



class ParticleMap(object):
    def __init__(self, ax, global_map, particle_list, values, target_particles=300,  draw_max=2000, resample_period=10, ):
        self.ax = ax
        self.draw_max = draw_max
        self.global_map = global_map
        self.particle_list = particle_list
        mcl.draw_map_state(global_map, particle_list, ax=self.ax, draw_max=self.draw_max)
        self.i = 1
        self.target_particles = target_particles
        self.resample_period = resample_period
        self.values = values
        self.bar = Bar('Processing', max=len(values), suffix='%(index)d/%(max)d - %(percent)d%%')


    def update(self, message):
        print(self.i)
        if self.i % self.resample_period == 0:# Resample and plot state
            self.particle_list = mcl.mcl_update(self.particle_list, message, resample=True,
                                                target_particles=self.target_particles) # Update
            plt.cla()
            mcl.draw_map_state(self.global_map, self.particle_list, self.ax, draw_max=self.draw_max)
            #print(pd.Series([p.weight for p in self.particle_list]).describe())
        else: # Just update particle weights / locations - do not resample
            self.particle_list = mcl.mcl_update(self.particle_list, message,
                                                target_particles=self.target_particles) # Update
        self.i += 1
        self.bar.next()
        if self.i == len(self.values) + 1:
            self.bar.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', default='./mcl_test_arena.mp4')
    parser.add_argument('--log', default='data/log/test_arena.dat')
    parser.add_argument('--map', default='data/map/test_arena.dat')
    parser.add_argument('--range', default='data/test_arena_range_array_120bin.npy')

    args = parser.parse_args()

    main(video_file=args.video, log_file=args.log, map_file=args.map, range_file=args.range)
