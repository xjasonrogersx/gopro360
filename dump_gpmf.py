from py_gpmf_parser.gopro_telemetry_extractor import GoProTelemetryExtractor

file_path = "GS010014-eqr-prores-telemetry.mov"
extractor = GoProTelemetryExtractor(file_path)

try:
    extractor.open_source()
    print(f"=== Extraction Summary for {file_path} ===\n")

    # 1. Inspect extracted devices and stream keys if available internally
    if hasattr(extractor, "devices") and extractor.devices:
        for idx, dev in enumerate(extractor.devices):
            print(f"Device [{idx}]: {getattr(dev, 'name', 'Unknown')}")
            streams = getattr(dev, "streams", {})
            for key, stream in streams.items():
                data = getattr(stream, "data", [])
                units = getattr(stream, "units", "")
                count = len(data) if data is not None else 0
                print(f"  └─ Stream '{key}': {count} samples (Units: {units})")
        print("\n" + "=" * 50 + "\n")

    # 2. Check standard telemetry keys
    keys_to_check = ["ACCL", "GYRO", "GPS5", "GPS9", "GRAV", "CORI", "SHUT", "ISOE"]

    for key in keys_to_check:
        try:
            data, timestamps = extractor.extract_data(key)
            if data is not None and len(data) > 0:
                print(f"✅ Found '{key}' stream:")
                print(f"   • Total Samples : {len(data)}")
                print(f"   • Time Range    : {timestamps[0]:.4f}s to {timestamps[-1]:.4f}s")
                print(f"   • First Sample  : {data[0]}")
                print(f"   • Last Sample   : {data[-1]}")
                # print(f"   • Full Data     : {data}\n")
            else:
                print(f"⚠️  Stream '{key}': Present in schema but contains 0 samples.")
        except Exception as e:
            # Key not found in GPMF payload
            pass

finally:
    extractor.close_source()