# app.py - interfaz gráfica (CustomTkinter + matplotlib)
import json
import os
import threading

import customtkinter as ctk
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, LogFormatter

import config
import trainer
from core.banco_filtros import (diferencias_absolutas, energias_subbandas,
                               espectro, espectro_promedio, frecuencias,
                               indices_de_corte, suma_diferencias_absolutas)

ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")


def crear_grafica(master, figsize=(7, 4)):
    """Crea figura, eje, lienzo y barra de herramientas de Matplotlib.

    La barra trae los botones de inicio, atrás/adelante, desplazar (mano),
    zoom por rectángulo (lupa) y guardar imagen.
    """
    fig = Figure(figsize=figsize, dpi=100)
    ax = fig.add_subplot(111)
    contenedor = ctk.CTkFrame(master, fg_color="white")
    contenedor.pack(fill="both", expand=True, padx=5, pady=5)
    canvas = FigureCanvasTkAgg(fig, master=contenedor)
    toolbar = NavigationToolbar2Tk(canvas, contenedor, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    activar_zoom_rueda(canvas, ax, toolbar)
    return fig, ax, canvas, toolbar


def formato_hz(ax):
    """En el eje x logarítmico muestra 100, 1000, 10000 en lugar de 10^2...

    Las marcas intermedias (200, 300, ...) solo se rotulan al hacer zoom.
    """
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))
    ax.xaxis.set_minor_formatter(LogFormatter(labelOnlyBase=False,
                                              minor_thresholds=(1, 0.4)))


def _zoom_eje(lim, centro, factor, escala_log):
    """Acerca (factor < 1) o aleja (factor > 1) un eje alrededor de centro."""
    if escala_log:
        a, b, c = np.log10(lim[0]), np.log10(lim[1]), np.log10(centro)
        return 10 ** (c - (c - a) * factor), 10 ** (c + (b - c) * factor)
    a, b = lim
    return centro - (centro - a) * factor, centro + (b - centro) * factor


def activar_zoom_rueda(canvas, ax, toolbar, paso=1.25):
    """Rueda del mouse: zoom centrado en el cursor.

    Rueda sola: zoom en ambos ejes. Ctrl + rueda: solo eje x.
    Shift + rueda: solo eje y. El botón "inicio" de la barra vuelve a la
    vista completa.
    """
    def al_girar(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        factor = 1 / paso if event.button == "up" else paso
        toolbar.push_current()  # para que "atrás" deshaga el zoom
        tecla = event.key or ""
        if "shift" not in tecla:
            ax.set_xlim(_zoom_eje(ax.get_xlim(), event.xdata, factor,
                                  ax.get_xscale() == "log"))
        if "control" not in tecla and "ctrl" not in tecla:
            ax.set_ylim(_zoom_eje(ax.get_ylim(), event.ydata, factor,
                                  ax.get_yscale() == "log"))
        canvas.draw_idle()

    canvas.mpl_connect("scroll_event", al_girar)


# ---------- Tablas de resultados ----------
AZUL = "#1F6AA5"
FILA_PAR, FILA_IMPAR = "#FFFFFF", "#F2F5F9"
FILA_TOTAL = "#E3EAF2"
VERDE_FONDO, VERDE_TEXTO = "#DCF2E3", "#1E7B3A"


def rangos_subbandas(cortes_hz, fs):
    """Textos 'a – b' de cada subbanda a partir de los cortes en Hz."""
    lim = [0] + sorted(cortes_hz) + [fs / 2]
    return ["%g – %g" % (lim[i], lim[i + 1]) for i in range(len(lim) - 1)]


class TablaResultados(ctk.CTkFrame):
    """Tabla simple con cabecera, filas alternadas, fila total opcional
    y una columna resaltada (la de la clase ganadora)."""

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#C9D3DE", corner_radius=6, **kwargs)

    def mostrar(self, cabeceras, filas, total=None, resaltar_col=None):
        for w in self.winfo_children():
            w.destroy()
        normal = ctk.CTkFont(size=13)
        negrita = ctk.CTkFont(size=13, weight="bold")

        def celda(texto, fila, col, fondo, color="#1A1A1A", fuente=normal):
            ctk.CTkLabel(self, text=texto, fg_color=fondo, text_color=color,
                         font=fuente, corner_radius=0, height=30
                         ).grid(row=fila, column=col, sticky="nsew",
                                padx=(0 if col else 1, 1), pady=(0 if fila else 1, 1))

        for c, texto in enumerate(cabeceras):
            celda(texto, 0, c, AZUL, "white", negrita)
            self.grid_columnconfigure(c, weight=1, uniform="col")
        for f, fila in enumerate(filas, start=1):
            fondo = FILA_PAR if f % 2 else FILA_IMPAR
            for c, texto in enumerate(fila):
                if c == resaltar_col:
                    celda(texto, f, c, VERDE_FONDO, VERDE_TEXTO)
                else:
                    celda(texto, f, c, fondo)
        if total is not None:
            f = len(filas) + 1
            for c, texto in enumerate(total):
                if c == resaltar_col:
                    celda(texto, f, c, VERDE_FONDO, VERDE_TEXTO, negrita)
                else:
                    celda(texto, f, c, FILA_TOTAL, fuente=negrita)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Reconocimiento de género por banco de filtros")
        self.geometry("1150x850")

        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)
        self.tab_ent = self.tabview.add("Entrenamiento")
        self.tab_rec = self.tabview.add("Reconocimiento")

        # Estado compartido
        self.frec = None
        self.promedios = None
        self.datos = None
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

        # Tabla de resultados del entrenamiento (abajo; se crea antes que la
        # gráfica para que siempre tenga su espacio)
        self.tabla_ent = TablaResultados(frame)
        self.tabla_ent.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        ctk.CTkLabel(self.tabla_ent, text="Carga los espectros, marca los cortes y "
                     "presiona «Entrenar y guardar» para ver los vectores de energía.",
                     text_color="#555555").pack(padx=20, pady=20)
        self.lbl_info_ent = ctk.CTkLabel(frame, text="", text_color="#555555",
                                         anchor="w")
        self.lbl_info_ent.pack(side="bottom", fill="x", padx=8)

        # Figura con barra de zoom
        (self.fig_ent, self.ax_ent, self.canvas_ent,
         self.toolbar_ent) = crear_grafica(frame)
        self.fig_ent.canvas.mpl_connect("button_press_event", self.click_corte)

    def cargar_espectros(self):
        datos = trainer.espectros_por_clase()
        self.datos = datos
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
        self.ax_ent.set_xscale("log")
        self.ax_ent.set_xlim(40, config.FS / 2)
        formato_hz(self.ax_ent)
        self.ax_ent.set_xlabel("Frecuencia (Hz, escala logarítmica)")
        self.ax_ent.set_ylabel("Magnitud |X(k)|")
        self.ax_ent.set_title("Espectros promedio por clase (clic para marcar cortes)")
        self.ax_ent.legend()
        self.ax_ent.grid(True)
        self.canvas_ent.draw()
        self.toolbar_ent.update()  # la vista actual pasa a ser la de "inicio"
        self.lbl_cortes_ent.configure(text="Cortes: " + str(self.cortes_hz))

    def click_corte(self, event):
        if event.xdata is None or self.frec is None:
            return
        # Con la lupa o la mano activas el clic es para el zoom, no un corte
        if self.toolbar_ent.mode or event.button != 1:
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
        energias = trainer.vectores_por_clase(self.datos, self.indices)

        os.makedirs(os.path.dirname(config.RUTA_MODELO), exist_ok=True)
        modelo = {
            "fs": config.FS,
            "N": config.N,
            "cortes_hz": list(self.cortes_hz),
            "energias": energias,
        }
        with open(config.RUTA_MODELO, "w", encoding="utf-8") as f:
            json.dump(modelo, f, indent=2, ensure_ascii=False)

        # Tabla: una fila por subbanda y una columna de energía por clase
        rangos = rangos_subbandas(self.cortes_hz, config.FS)
        lim = [0] + list(self.indices) + [len(self.frec)]
        cabeceras = ["Subbanda", "Rango (Hz)", "Índices k"] + [
            "E_%s" % c for c in config.CLASES]
        filas = []
        for i in range(len(lim) - 1):
            filas.append([str(i + 1), rangos[i],
                          "%d – %d" % (lim[i], lim[i + 1] - 1)] +
                         ["%.4f" % energias[c][i] for c in config.CLASES])
        self.tabla_ent.mostrar(cabeceras, filas)
        self.lbl_info_ent.configure(
            text="Modelo guardado en %s  ·  %d subbandas  ·  fs = %d Hz  ·  N = %d"
            % (config.RUTA_MODELO, len(filas), config.FS, config.N))

    # ---------- Pestaña Reconocimiento ----------
    def _construir_reconocimiento(self):
        frame = self.tab_rec
        controles = ctk.CTkFrame(frame)
        controles.pack(fill="x", padx=5, pady=5)

        ctk.CTkButton(controles, text="Grabar y reconocer",
                      command=self.grabar_reconocer).pack(side="left", padx=5)
        self.lbl_estado = ctk.CTkLabel(controles, text="Listo")
        self.lbl_estado.pack(side="left", padx=10)


        # Panel de resultados: tabla a la izquierda, veredicto a la derecha
        # (se crea antes que la gráfica para que siempre tenga su espacio)
        panel = ctk.CTkFrame(frame, fg_color="transparent")
        panel.pack(side="bottom", fill="x", padx=5, pady=(0, 5))
        self.tabla_rec = TablaResultados(panel)
        self.tabla_rec.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ctk.CTkLabel(self.tabla_rec, text="Presiona «Grabar y reconocer» para ver "
                     "las energías y diferencias por subbanda.",
                     text_color="#555555").pack(padx=20, pady=30)

        tarjeta = ctk.CTkFrame(panel, width=260, corner_radius=10,
                               fg_color="white", border_width=2,
                               border_color="#C9D3DE")
        tarjeta.pack(side="right", fill="y")
        tarjeta.pack_propagate(False)
        ctk.CTkLabel(tarjeta, text="Género reconocido", text_color="#555555",
                     font=ctk.CTkFont(size=14)).pack(pady=(22, 2))
        self.lbl_genero = ctk.CTkLabel(tarjeta, text="—", text_color=AZUL,
                                       font=ctk.CTkFont(size=34, weight="bold"))
        self.lbl_genero.pack(pady=4)
        self.lbl_detalle = ctk.CTkLabel(tarjeta, text="", text_color="#555555",
                                        font=ctk.CTkFont(size=12), justify="center")
        self.lbl_detalle.pack(pady=(2, 18))

        (self.fig_rec, self.ax_rec, self.canvas_rec,
         self.toolbar_rec) = crear_grafica(frame)

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
            E_X = trainer.vector_energias(z, indices)

            energias_clases = modelo.get("energias", {})
            resultados = []
            for clase, E_C in energias_clases.items():
                diffs = diferencias_absolutas(E_X, E_C)
                suma = suma_diferencias_absolutas(E_X, E_C)
                resultados.append((clase, [float(d) for d in diffs], float(suma)))
            resultados.sort(key=lambda r: r[2])

            ganador = resultados[0][0] if resultados else "?"
            referencia = self._referencia_escalada(ganador, z)

            # Actualizar UI en el hilo principal
            self.after(0, lambda: self._mostrar_reconocimiento(
                z, frec, cortes_hz, E_X, resultados, ganador, referencia))
            self.after(0, lambda: self.lbl_estado.configure(text="Listo"))
        except Exception as e:
            self.after(0, lambda: self.lbl_estado.configure(text="Error: %s" % e))

    def _referencia_escalada(self, clase, z):
        """Espectro promedio de la clase, escalado a la energía de z.

        Las grabaciones del banco están normalizadas y la del micrófono no,
        así que el promedio se multiplica por un factor que iguala la energía
        total (suma de |X(k)|^2) de ambos espectros. Así se compara la forma,
        que es lo que mide el vector de energías relativo.
        Devuelve None si no hay grabaciones para calcular el promedio.
        """
        try:
            if self.promedios is None or clase not in self.promedios:
                self.after(0, lambda: self.lbl_estado.configure(
                    text="Cargando espectro promedio de %s..." % clase))
                self.datos = trainer.espectros_por_clase()
                self.promedios = {c: espectro_promedio(self.datos[c])
                                  for c in config.CLASES}
            ref = np.asarray(self.promedios[clase], dtype=float)
            if ref.size != np.size(z):
                return None
            energia_ref = float(np.sum(ref ** 2))
            if energia_ref <= 0:
                return None
            return ref * np.sqrt(float(np.sum(np.asarray(z) ** 2)) / energia_ref)
        except Exception:
            return None

    def _mostrar_reconocimiento(self, z, frec, cortes_hz, E_X, resultados,
                                ganador, referencia=None):
        self.ax_rec.clear()
        if referencia is not None:
            # Promedio de la clase reconocida, de fondo y en gris
            self.ax_rec.plot(frec, referencia, color="#9E9E9E", linewidth=1.2,
                             alpha=0.8, zorder=1,
                             label="promedio %s (escalado)" % ganador)
        self.ax_rec.plot(frec, z, label="señal", zorder=2)
        for fc in cortes_hz:
            self.ax_rec.axvline(fc, color="gray", linestyle="--")
        self.ax_rec.set_xscale("log")
        self.ax_rec.set_xlim(40, frec[-1])
        formato_hz(self.ax_rec)
        self.ax_rec.set_xlabel("Frecuencia (Hz, escala logarítmica)")
        self.ax_rec.set_ylabel("Magnitud |X(k)|")
        if referencia is not None:
            self.ax_rec.set_title("Espectro de la señal grabada vs. promedio de %s"
                                  % ganador)
        else:
            self.ax_rec.set_title("Espectro de la señal grabada")
        self.ax_rec.legend()
        self.ax_rec.grid(True)
        self.canvas_rec.draw()
        self.toolbar_rec.update()

        # Tabla: una fila por subbanda y una columna |E_X - E_C| por clase
        por_clase = {clase: (diffs, suma) for clase, diffs, suma in resultados}
        clases = [c for c in config.CLASES if c in por_clase]
        rangos = rangos_subbandas(cortes_hz, frec[-1] * 2)
        cabeceras = ["Subbanda", "Rango (Hz)", "E_X"] + [
            "|E_X − E_%s|" % c for c in clases]
        filas = []
        for i, E in enumerate(E_X):
            filas.append([str(i + 1), rangos[i], "%.4f" % E] +
                         ["%.4f" % por_clase[c][0][i] for c in clases])
        total = ["Suma", "", "%.4f" % float(np.sum(E_X))] + [
            "%.4f" % por_clase[c][1] for c in clases]
        col = 3 + clases.index(ganador) if ganador in clases else None
        self.tabla_rec.mostrar(cabeceras, filas, total, resaltar_col=col)

        # Tarjeta con el veredicto
        self.lbl_genero.configure(text=ganador.upper(), text_color=VERDE_TEXTO)
        detalle = ["menor suma de diferencias"]
        detalle += ["%s: %.4f" % (c, por_clase[c][1]) for c in clases]
        self.lbl_detalle.configure(text="\n".join(detalle))


if __name__ == "__main__":
    app = App()
    app.mainloop()
