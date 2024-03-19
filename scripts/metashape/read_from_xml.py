import xml.etree.ElementTree as ET

# Specify the path to your XML file
xml_file_path = r'C:\Users\threedom\Desktop\IMC24\aerial_vs_terrestrial_church\metashape\prova\agisoft.xml'

# Parse the XML file
tree = ET.parse(xml_file_path)
root = tree.getroot()

for camera in root.iter('camera'):
    camera_id = camera.get('id')
    sensor_id = camera.get('sensor_id')
    component_id = camera.get('component_id')
    label = camera.get('label')
    #print(f"Camera ID: {camera_id}, Sensor ID: {sensor_id}, Component ID: {component_id}, Label: {label}")

    # Access transform within each camera
    transform_element = camera.find('transform')
    if transform_element is not None:
        transform_text = transform_element.text
        values_list = transform_text.split()
        import numpy as np
        transf_matrix = (np.array([float(value) for value in values_list])).reshape((4,4))
        R = transf_matrix[:3,:3]
        t = transf_matrix[:3,3]
        print(label, t[0], t[1], t[2])