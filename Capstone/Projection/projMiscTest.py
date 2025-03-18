# from PyQt6.QtGui import QImage

# formats = [attr for attr in dir(QImage) if attr.startswith("Format_")]
# print("Available QImage Formats in PyQt6:")
# for f in formats:
#     print(f)

# from PyQt6.QtGui import QImage
# # print([attr for attr in dir(QImage) if attr.startswith("Format_")])
# print(dir(QImage))

from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID

# Get a list of all visible windows
window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)

print("Active windows:")
for window in window_list:
    window_title = window.get("kCGWindowName", "No Name")
    owner_name = window.get("kCGWindowOwnerName", "Unknown App")
    print(f"Title: {window_title} | App: {owner_name}")


