unsigned int devices_count {};

nvmlInit ();
nvmlDeviceGetCount (&devices_count);

nvmlDevice_t device;
nvmlDeviceGetHandleByIndex (0, &device);

nvmlNvLinkUtilizationControl_t utilization_control;
utilization_control.units = NVML_NVLINK_COUNTER_UNIT_BYTES;
utilization_control.pktfilter = NVML_NVLINK_COUNTER_PKTFILTER_ALL;
nvmlDeviceFreezeNvLinkUtilizationCounter (device, 0, 0, NVML_FEATURE_DISABLED);
nvmlDeviceSetNvLinkUtilizationControl (device, 0, 0, &utilization_control, 1);

unsigned long long int tx_before {};
unsigned long long int rx_before {};
nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_before, &tx_before);

// code to measure

unsigned long long int tx_after {};
unsigned long long int rx_after {};
nvmlDeviceGetNvLinkUtilizationCounter (device, 0, 0, &rx_after, &tx_after);

const unsigned long long int tx = tx_after - tx_before;
const unsigned long long int rx = rx_after - rx_before;
