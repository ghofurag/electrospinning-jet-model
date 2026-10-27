# ==============================================================================
# Program: Simulasi berdasarkan Jarak (Versi Lengkap)
# ==============================================================================

import numpy as np
import plotly.graph_objects as go
import time

# --- Bagian 1: Definisi Parameter Fisika dan Konstanta ---
k = 8.9875e9          
epsilon_0 = 8.854e-12 

alpha = 41e-1          
V = 20e3               
viskositas = 0.375e-3       
G = 50               
a0 = 3e-6           
rho = 1021.95             
d = 10e-2             
D = 1.12e-10              
epsilon = 1e-5       
L0 = 1e-4             
q_volume = 1e-3      

# --- Bagian 2: Diskretisasi dan Persamaan Gerak (Representasi Node) ---
def hitung_muatan_dan_massa(panjang_node):
    muatan = q_volume * np.pi * a0**2 * panjang_node
    massa = rho * np.pi * a0**2 * panjang_node
    return muatan, massa

class Node:
    def __init__(self, x, y, z, id):
        self.r = np.array([x, y, z], dtype=float)
        self.v = np.zeros(3, dtype=float)
        self.a = np.zeros(3, dtype=float)
        self.muatan = 0.0
        self.massa = 0.0
        self.radius = a0
        self.sigma = 0.0
        self.id = id

# --- Bagian 3: Fungsi-fungsi Gaya ---
def calculate_forces(nodes, d, V, k, alpha, G, viskositas, dt):
    num_nodes = len(nodes)
    medan_listrik = V / d
    for i in range(num_nodes):
        F_coulomb = np.zeros(3)
        for j in range(num_nodes):
            if i != j:
                r_ij = nodes[i].r - nodes[j].r
                jarak = np.linalg.norm(r_ij)
                if jarak > 1e-9: 
                    F_coulomb += k * nodes[i].muatan * nodes[j].muatan * r_ij / jarak**3
        F_medan_listrik = nodes[i].muatan * medan_listrik * np.array([1, 0, 0])
        F_visco = np.zeros(3)
        F_permukaan = np.zeros(3)
        if i > 0:
            r_i_prev = nodes[i].r - nodes[i-1].r
            jarak_prev = np.linalg.norm(r_i_prev)
            nodes[i].sigma = G / viskositas
            F_visco -= np.pi * nodes[i-1].radius**2 * nodes[i].sigma * r_i_prev / jarak_prev
            F_permukaan -= np.pi * nodes[i-1].radius * alpha * r_i_prev / jarak_prev
        if i < num_nodes - 1:
            r_i_next = nodes[i+1].r - nodes[i].r
            jarak_next = np.linalg.norm(r_i_next)
            F_visco += np.pi * nodes[i].radius**2 * nodes[i+1].sigma * r_i_next / jarak_next
            F_permukaan += np.pi * nodes[i].radius * alpha * r_i_next / jarak_next
        resultan_gaya = F_coulomb + F_medan_listrik + F_visco + F_permukaan
        if nodes[i].massa > 0:
            nodes[i].a = resultan_gaya / nodes[i].massa

# --- Bagian 4: Simulasi Utama (Loop Berdasarkan Jarak) ---
def run_simulation_by_distance(d, V, k, alpha, G, viskositas, D, L0, a0, dt):
    nodes = []
    node_trajectories = {}
    all_diameters = []
    r_tip = np.zeros(3)
    waktu_keluar_node_berikutnya = 0
    node_id_counter = 0
    time = 0
    
    # Kondisi berhenti: loop berjalan selama node pertama belum mencapai kolektor
    while (len(nodes) == 0) or (nodes[0].r[0] < d):
        if not nodes or (np.linalg.norm(nodes[-1].r - r_tip) >= L0):
            if node_id_counter == 0:
                theta = np.random.uniform(0, 2 * np.pi)
                y = epsilon * np.sin(theta)
                z = epsilon * np.cos(theta)
                x = 0
            else:
                x, y, z = 0, 0, 0
            
            node_baru = Node(x, y, z, node_id_counter)
            node_baru.muatan, node_baru.massa = hitung_muatan_dan_massa(L0)
            node_baru.v[0] = D / (np.pi * a0**2)
            
            nodes.append(node_baru)
            node_trajectories[node_baru.id] = []
            node_id_counter += 1
            
            waktu_keluar_node_berikutnya = time + L0 / node_baru.v[0]
            
        if nodes:
            calculate_forces(nodes, d, V, k, alpha, G, viskositas, dt)
            
            for i, node in enumerate(nodes):
                node.v += node.a * dt
                node.r += node.v * dt
                node_trajectories[node.id].append(node.r.copy())
                
                if i > 0:
                    current_length = np.linalg.norm(node.r - nodes[i-1].r)
                    stretching_factor = current_length / L0
                    current_radius = a0 / np.sqrt(stretching_factor)
                    current_diameter = 2 * current_radius
                    all_diameters.append(current_diameter)
        
        time += dt
        
    return node_trajectories, all_diameters, time

# --- Bagian 5: Visualisasi dengan Plotly ---
def visualize_plotly(waktu_simulasi, node_trajectories, d, avg_diameter, filename):
    fig = go.Figure()
    
    first_node_trajectory = node_trajectories.get(0, [])
    
    if first_node_trajectory:
        trajectory = np.array(first_node_trajectory)
        fig.add_trace(go.Scatter3d(
            x=trajectory[:,0], y=trajectory[:,1], z=trajectory[:,2],
            mode='lines',
            line=dict(color='blue', width=4),  # Saya pertebal garisnya juga menjadi 4
            name='Lintasan Jet'
        ))
    
    # --- PENGATURAN FONT DAN LAYOUT ---
    fig.update_layout(
        # Pengaturan Font Global (Default)
        font=dict(
            family="Arial, sans-serif",
            size=18,  # Ukuran dasar teks
            color="black"
        ),
        
        # Pengaturan Judul
        title=dict(
            text=f"<b>Electrospinning Jet Trajectory Simulation</b><br>"
                 f"<span style='font-size: 18px'>Average Fiber Diameter: {avg_diameter:.6e} m</span><br>"
                 f"<span style='font-size: 18px'>Time to Collector: {waktu_simulasi:.6f} s</span>",
            y=0.95, # Posisi vertikal judul
            x=0.5,  # Posisi horizontal (tengah)
            xanchor='center',
            yanchor='top',
            font=dict(
                size=24, # Ukuran Font Judul Utama
                color="black"
            )
        ),
        
        # Pengaturan Sumbu 3D (Scene)
        scene=dict(
            # Sumbu X
            xaxis=dict(
                title=dict(text='<b>X (m) [Trajectory]</b>', font=dict(size=18)), # Judul Sumbu Tebal & Besar
                tickfont=dict(size=16, color='black'), # Angka di sumbu
                range=[0, d],
                backgroundcolor="rgb(240, 240, 240)", # Sedikit abu-abu agar grid terlihat
                gridcolor="white",
                showbackground=True,
                nticks=10,
            ),
            # Sumbu Y
            yaxis=dict(
                title=dict(text='<b>Y (m)</b>', font=dict(size=18)),
                tickfont=dict(size=16, color='black'),
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
            ),
            # Sumbu Z
            zaxis=dict(
                title=dict(text='<b>Z (m)</b>', font=dict(size=18)),
                tickfont=dict(size=16, color='black'),
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
            ),
        ),
        
        autosize=False,
        width=1000,  # Lebar diperbesar sedikit
        height=900,  # Tinggi diperbesar sedikit
        margin=dict(l=50, r=50, b=50, t=65) # Margin disesuaikan agar judul tidak terpotong
    )

    # Simpan atau tampilkan (sesuaikan dengan kebutuhan script Anda)
    fig.show() 
    fig.write_html(filename) 
    return fig

# ==============================================================================
# Jalankan Simulasi
# ==============================================================================
if __name__ == '__main__':
    dt = 5e-6
    output_filename = "simulasi_electrospinning.html"

    print("Starting Program...")
    start_time = time.time()
    lintasan_node, all_diameters, waktu_simulasi = run_simulation_by_distance(d, V, k, alpha, G, viskositas, D, L0, a0, dt)
    end_time = time.time()
    
    avg_diameter = np.mean(all_diameters) if all_diameters else 0
    
    print(f"Simulation complete. Time taken for the first node to reach the collector: {waktu_simulasi:.6f} seconds")
    print(f"Computation time: {end_time - start_time:.2f} seconds")
    print(f"Average generated fiber diameter: {avg_diameter:.6e} m")
    
    visualize_plotly(waktu_simulasi,lintasan_node, d, avg_diameter, output_filename)
    
    #print(f"Plot interaktif telah disimpan ke file: {output_filename}")
