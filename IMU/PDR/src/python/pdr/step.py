import numpy as np
import scipy.signal as signal

class Step:
    # Peak detection for step detection for provided windows in acc_data..
    # Parameter acc_data contains lists of 4 element: 1st is timestamp, the remaining are accelerator axes.
    def peak_detection(acc_data, threshold, lag = 1, influence = 0.5):
        y = Step.__acc_data2z_axis(acc_data)
        signals = np.zeros(len(y))
        filteredY = np.array(y)
        avgFilter = [0]*len(y)
        stdFilter = [0]*len(y)
        avgFilter[lag - 1] = np.mean(y[0:lag])
        stdFilter[lag - 1] = np.std(y[0:lag])
        for i in range(lag, len(y)):
            if abs(y[i] - avgFilter[i-1]) > threshold * stdFilter [i-1]:
                if y[i] > avgFilter[i-1]:
                     signals[i] = 1
                else:
                     signals[i] = -1

                filteredY[i] = influence * y[i] + (1 - influence) * filteredY[i-1]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])
            else:
                signals[i] = 0
                filteredY[i] = y[i]
                avgFilter[i] = np.mean(filteredY[(i-lag+1):i+1])
                stdFilter[i] = np.std(filteredY[(i-lag+1):i+1])

        return dict(signals = np.asarray(signals),
                    avgFilter = np.asarray(avgFilter),
                    stdFilter = np.asarray(stdFilter))

    # List of acceleration data to list of Z-axis acceleration data.
    @staticmethod
    def __acc_data2z_axis(acc_data):
        a_axis_data = []

        for data in acc_data:
            a_axis_data.append(data[3])

        return a_axis_data

    # Step detection provided by Kaggle>
    # https://github.com/location-competition/indoor-location-competition-20/blob/master/compute_f.py#L210
    @staticmethod
    def kaggle_detect(acce_datas):
        step_timestamps = np.array([])
        step_indexs = np.array([], dtype=int)
        step_acce_max_mins = np.zeros((0, 4))
        sample_freq = 50
        window_size = 22
        low_acce_mag = 0.6
        step_criterion = 1
        interval_threshold = 250

        acce_max = np.zeros((2,))
        acce_min = np.zeros((2,))
        acce_binarys = np.zeros((window_size,), dtype=int)
        acce_mag_pre = 0
        state_flag = 0

        warmup_data = np.ones((window_size,)) * 9.81
        filter_b, filter_a, filter_zf = Step.__init_parameters_filter(sample_freq, warmup_data)
        acce_mag_window = np.zeros((window_size, 1))

        # Detect steps according to acceleration magnitudes
        for i in np.arange(0, np.size(acce_datas, 0)):
            acce_data = acce_datas[i, :]
            acce_mag = np.sqrt(np.sum(acce_data[1:] ** 2))

            acce_mag_filt, filter_zf = signal.lfilter(filter_b, filter_a, [acce_mag], zi=filter_zf)
            acce_mag_filt = acce_mag_filt[0]

            acce_mag_window = np.append(acce_mag_window, [acce_mag_filt])
            acce_mag_window = np.delete(acce_mag_window, 0)
            mean_gravity = np.mean(acce_mag_window)
            acce_std = np.std(acce_mag_window)
            mag_threshold = np.max([low_acce_mag, 0.4 * acce_std])

            # Detect valid peak or valley of acceleration magnitudes
            acce_mag_filt_detrend = acce_mag_filt - mean_gravity
            if acce_mag_filt_detrend > np.max([acce_mag_pre, mag_threshold]):
                # peak
                acce_binarys = np.append(acce_binarys, [1])
                acce_binarys = np.delete(acce_binarys, 0)
            elif acce_mag_filt_detrend < np.min([acce_mag_pre, -mag_threshold]):
                # valley
                acce_binarys = np.append(acce_binarys, [-1])
                acce_binarys = np.delete(acce_binarys, 0)
            else:
                # between peak and valley
                acce_binarys = np.append(acce_binarys, [0])
                acce_binarys = np.delete(acce_binarys, 0)

            if (acce_binarys[-1] == 0) and (acce_binarys[-2] == 1):
                if state_flag == 0:
                    acce_max[:] = acce_data[0], acce_mag_filt
                    state_flag = 1
                elif (state_flag == 1) and ((acce_data[0] - acce_max[0]) <= interval_threshold) and (
                        acce_mag_filt > acce_max[1]):
                    acce_max[:] = acce_data[0], acce_mag_filt
                elif (state_flag == 2) and ((acce_data[0] - acce_max[0]) > interval_threshold):
                    acce_max[:] = acce_data[0], acce_mag_filt
                    state_flag = 1

            # choose reasonable step criterion and check if there is a valid step
            # save step acceleration data: step_acce_max_mins = [timestamp, max, min, variance]
            step_flag = False
            if step_criterion == 2:
                if (acce_binarys[-1] == -1) and ((acce_binarys[-2] == 1) or (acce_binarys[-2] == 0)):
                    step_flag = True
            elif step_criterion == 3:
                if (acce_binarys[-1] == -1) and (acce_binarys[-2] == 0) and (np.sum(acce_binarys[:-2]) > 1):
                    step_flag = True
            else:
                if (acce_binarys[-1] == 0) and acce_binarys[-2] == -1:
                    if (state_flag == 1) and ((acce_data[0] - acce_min[0]) > interval_threshold):
                        acce_min[:] = acce_data[0], acce_mag_filt
                        state_flag = 2
                        step_flag = True
                    elif (state_flag == 2) and ((acce_data[0] - acce_min[0]) <= interval_threshold) and (
                            acce_mag_filt < acce_min[1]):
                        acce_min[:] = acce_data[0], acce_mag_filt
            if step_flag:
                step_timestamps = np.append(step_timestamps, acce_data[0])
                step_indexs = np.append(step_indexs, [i])
                step_acce_max_mins = np.append(step_acce_max_mins,
                                           [[acce_data[0], acce_max[1], acce_min[1], acce_std ** 2]], axis=0)
            acce_mag_pre = acce_mag_filt_detrend

        return step_timestamps, step_indexs, step_acce_max_mins

    @staticmethod
    def __init_parameters_filter(sample_freq, warmup_data, cut_off_freq=2):
        order = 4
        filter_b, filter_a = signal.butter(order, cut_off_freq / (sample_freq / 2), 'low', False)
        zf = signal.lfilter_zi(filter_b, filter_a)
        _, zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)
        _, filter_zf = signal.lfilter(filter_b, filter_a, warmup_data, zi=zf)

        return filter_b, filter_a, filter_zf
