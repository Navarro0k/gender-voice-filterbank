# app.py - interfaz gráfica (CustomTkinter + matplotlib)
import json
import os
import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import config
import trainer
from core.banco_filtros import (diferencias_absolutas, energias_subbandas,
                               espectro, espectro_promedio, frecuencias,
                               indices_de_corte, suma_diferencias_absolutas)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de género por banco de filtros")
        self.geometry("900x700")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_ent = self.tabview.add("Entrenamiento")
        self.tab_rec = self.tabview.add("Reconocimiento")

        # Estado compartido
        self.frec = None
        self.promedios = None
        self.cortes_hz = list(config.CORTES_HZ)
        self.indices = None

        self._construir_entrenamiento()
        self._construir_reconocimiento()

    # ---------- Pestaña Entrenamiento ----------
    def _construir_entrenamiento(self):
        frame = self.tab_ent
        controles = ctk.CTkFrame(frame)
        controles.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(controles, text="Cargar espectros promedio",
                      command=self.cargar_espectros).pack(side="left", padx=5)
        ctk.CTkButton(controles, text="Limpiar cortes",
                      command=self.limpiar_cortes).pack(side="left", padx=5)
        ctk.CTkButton(controles, text="Entrenar y guardar",
                      command=self.entrenar).pack(side="left", padx=5)

        self.lbl_cortes_ent = ctk.CTkLabel(controles, text="Cortes: " + str(self.cortes_hz))
        self.lbl_cortes_ent.pack(side="left", padx=10)

        # Figura
        self.fig_ent = Figure(figsize=(7, 4), dpi=100)
        self.ax_ent = self.fig_ent.add_subplot(111)
        self.canvas_ent = FigureCanvasTkAgg(self.fig_ent, master=frame)
        self.canvas_ent.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)
        self.fig_ent.canvas.mpl_connect("button_press_event", self.click_corte)

        # Tabla de resultados
        self.txt_ent = ctk.CTkTextbox(frame, height=120)
        self.txt_ent.pack(fill="x", padx=5, pady=5)

    def cargar_espectros(self):
        datos = trainer.espectros_por_clase()
        self.promedios = {}
        for clase in config.CLASES:
            self.promedios[clase] = espectro_promedio(datos[clase])
        self.frec = frecuencias(config.N, config.FS)
        self.redibujar_ent()

    def redibujar_ent(self):
        self.ax_ent.clear()
        if self.frec is not None and self.promedios is not None:
            for clase in config.CLASES:
                self.ax_ent.plot(self.frec, self.promedios[clase], label=clase)
        for fc in self.cortes_hz:
            self.ax_ent.axvline(fc, color="gray", linestyle="--")
        self.ax_ent.set_xlabel("Frecuencia (Hz)")
        self.ax_ent.set_ylabel("Magnitud |X(k)|")
        self.ax_ent.set_title("Espectros promedio por clase (clic para marcar cortes)")
        self.ax_ent.legend()
        self.ax_ent.grid(True)
        self.canvas_ent.draw()
        self.lbl_cortes_ent.configure(text="Cortes: " + str(self.cortes_hz))

    def click_corte(self, event):
        if event.xdata is None or self.frec is None:
            return
        if len(self.cortes_hz) >= config.N_CORTES:
            return
        idx = int(np.argmin(np.abs(self.frec - event.xdata)))
        fc = float(self.frec[idx])
        if fc <= 0 or fc >= self.frec[-1]:
            return
        if fc in self.cortes_hz:
            return
        self.cortes_hz.append(fc)
        self.cortes_hz = sorted(set(self.cortes_hz))
        self.redibujar_ent()

    def limpiar_cortes(self):
        self.cortes_hz = []
        self.redibujar_ent()

    def entrenar(self):
        if self.promedios is None:
            self.cargar_espectros()
        self.frec = frecuencias(config.N, config.FS)
        self.indices = indices_de_corte(self.frec, self.cortes_hz)
        energias = {}
        for clase in config.CLASES:
            energias[clase] = energias_subbandas(self.promedios[clase], self.indices)

        os.makedirs(os.path.dirname(config.RUTA_MODELO), exist_ok=True)
        modelo = {
            "fs": config.FS,
            "N": config.N,
            "cortes_hz": list(self.cortes_hz),
            "energias": energias,
        }
        with open(config.RUTA_MODELO, "w", encoding="utf-8") as f:
            json.dump(modelo, f, indent=2, ensure_ascii=False)

        # Mostrar tabla
        self.txt_ent.delete("1.0", "end")
        lineas = ["Vectores de energía por clase:"]
        for clase in config.CLASES:
            lineas.append("%s: %s" % (clase, [round(v, 4) for v in energias[clase]]))
        lineas.append("Índices de corte: %s" % self.indices)
        lineas.append("Cortes (Hz): %s" % self.cortes_hz)
        self.txt_ent.insert("end", "\n".join(lineas))

    # ---------- Pestaña Reconocimiento ----------
    def _construir_reconocimiento(self):
        frame = self.tab_rec
        controles = ctk.CTkFrame(frame)
        controles.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(controles, text="Grabar y reconocer",
                      command=self.grabar_reconocer).pack(side="left", padx=5)
        self.lbl_estado = ctk.CTkLabel(controles, text="Listo")
        self.lbl_estado.pack(side="left", padx=10)

        self.fig_rec = Figure(figsize=(7, 4), dpi=100)
        self.ax_rec = self.fig_rec.add_subplot(111)
        self.canvas_rec = FigureCanvasTkAgg(self.fig_rec, master=frame)
        self.canvas_rec.get_tk_widget().pack(fill="both", expand=True, padx=5, pady=5)

        self.txt_rec = ctk.CTkTextbox(frame, height=160)
        self.txt_rec.pack(fill="x", padx=5, pady=5)

    def grabar_reconocer(self):
        self.lbl_estado.configure(text="Grabando...")
        self.update()
        threading.Thread(target=self._proceso_reconocimiento, daemon=True).start()

    def _proceso_reconocimiento(self):
        try:
            modelo = {}
            if os.path.exists(config.RUTA_MODELO):
                with open(config.RUTA_MODELO, "r", encoding="utf-8") as f:
                    modelo = json.load(f)
            else:
                modelo = {
                    "fs": config.FS,
                    "N": config.N,
                    "cortes_hz": self.cortes_hz,
                    "energias": {},
                }

            fs = modelo.get("fs", config.FS)
            N = modelo.get("N", int(fs * config.DURACION))
            cortes_hz = modelo.get("cortes_hz", self.cortes_hz)

            audio = sd.rec(N, samplerate=fs, channels=1, dtype="float64")
            sd.wait()
            x = audio.flatten()

            z = espectro(x)
            frec = frecuencias(N, fs)
            indices = indices_de_corte(frec, cortes_hz)
            E_X = energias_subbandas(z, indices)

            energias_clases = modelo.get("energias", {})
            resultados = []
            for clase, E_C in energias_clases.items():
                diffs = diferencias_absolutas(E_X, E_C)
                suma = suma_diferencias_absolutas(E_X, E_C)
                resultados.append((clase, [float(d) for d in diffs], float(suma)))
            resultados.sort(key=lambda r: r[2])

            ganador = resultados[0][0] if resultados else "?"

            # Actualizar UI en el hilo principal
            self.after(0, lambda: self._mostrar_reconocimiento(
                z, frec, cortes_hz, E_X, resultados, ganador))
            self.after(0, lambda: self.lbl_estado.configure(text="Listo"))
        except Exception as e:
            self.after(0, lambda: self.lbl_estado.configure(text="Error: %s" % e))

    def _mostrar_reconocimiento(self, z, frec, cortes_hz, E_X, resultados, ganador):
        self.ax_rec.clear()
        self.ax_rec.plot(frec, z, label="señal")
        for fc in cortes_hz:
            self.ax_rec.axvline(fc, color="gray", linestyle="--")
        self.ax_rec.set_xlabel("Frecuencia (Hz)")
        self.ax_rec.set_ylabel("Magnitud |X(k)|")
        self.ax_rec.set_title("Espectro de la señal grabada")
        self.ax_rec.legend()
        self.ax_rec.grid(True)
        self.canvas_rec.draw()

        self.txt_rec.delete("1.0", "end")
        lineas = ["E_X: %s" % [round(v, 4) for v in E_X]]
        lineas.append("Diferencias por subbanda:")
        for clase, diffs, suma in resultados:
            lineas.append("  %s: diffs=%s  suma=%.4f" % (clase, [round(d, 4) for d in diffs], suma))
        lineas.append("")
        lineas.append("Género reconocido: %s" % ganador.upper())
        self.txt_rec.insert("end", "\n".join(lineas))


if __name__ == "__main__":
    app = App()
    app.mainloop()
