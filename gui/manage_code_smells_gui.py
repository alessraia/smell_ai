import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Optional

from llm_detection.catalog_service import CatalogValidationError, LLMCatalogService

class ManageCodeSmellsGUI:
    def __init__(self, master: tk.Tk, catalog_service: Optional[LLMCatalogService] = None):
        self.master = master
        self.catalog_service = catalog_service or LLMCatalogService()

        # State
        self._smell_display_to_id: dict[str, str] = {}
        self._current_smell_id: Optional[str] = None

        self._build_ui()
        self._load_smells_into_dropdown()

        # Handle window close
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.master.title("Gestione Code Smells (UC 1.3)")
        self.master.geometry("800x600")

        # --- Top Bar: Selection and Management ---
        top = ttk.Frame(self.master)
        top.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.master.grid_columnconfigure(0, weight=1)

        ttk.Label(top, text="Code smell:").grid(row=0, column=0, sticky="e", padx=(0, 6))

        self._smell_combo = ttk.Combobox(top, state="readonly", width=50)
        self._smell_combo.grid(row=0, column=1, sticky="w")
        self._smell_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_smell_selected())

        # Add (+) and Remove (-) buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=0, column=2, sticky="w", padx=(6, 0))

        self._add_btn = ttk.Button(btn_frame, text="+", width=3, command=self._on_add_smell)
        self._add_btn.pack(side="left", padx=2)
        
        self._remove_btn = ttk.Button(btn_frame, text="-", width=3, command=self._on_remove_smell)
        self._remove_btn.pack(side="left", padx=2)

        # --- Details Section ---
        details_frame = ttk.LabelFrame(self.master, text="Dettagli Code Smell")
        details_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        self.master.grid_rowconfigure(1, weight=1)
        details_frame.grid_columnconfigure(1, weight=1)
        details_frame.grid_rowconfigure(1, weight=1) # Description expands

        # Name (Read-only for existing)
        ttk.Label(details_frame, text="Nome:").grid(row=0, column=0, sticky="ne", padx=10, pady=10)
        self._name_var = tk.StringVar()
        self._name_entry = ttk.Entry(details_frame, textvariable=self._name_var, state="readonly")
        self._name_entry.grid(row=0, column=1, sticky="ew", padx=10, pady=10)

        # Description
        ttk.Label(details_frame, text="Descrizione:").grid(row=1, column=0, sticky="ne", padx=10, pady=0)
        self._desc_text = ScrolledText(details_frame, height=15, wrap="word")
        self._desc_text.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0, 10))

        # --- Bottom Bar: Actions ---
        actions = ttk.Frame(self.master)
        actions.grid(row=2, column=0, sticky="ew", padx=10, pady=10)
        actions.grid_columnconfigure(0, weight=1)

        self._save_btn = ttk.Button(actions, text="Salva Modifiche Descrizione", command=self._on_save_changes)
        self._save_btn.grid(row=0, column=0, sticky="w")

        self._exit_btn = ttk.Button(actions, text="Exit", command=self._on_close)
        self._exit_btn.grid(row=0, column=1, sticky="e")

    def _load_smells_into_dropdown(self):
        try:
            catalog = self.catalog_service.load()
        except Exception as e:
            messagebox.showerror("Errore Caricamento", f"Impossibile caricare il catalogo: {e}")
            return

        self._smell_display_to_id.clear()
        values = []

        # Sort by display name
        sorted_smells = sorted(catalog.smells, key=lambda s: s.display_name.lower())
        
        for smell in sorted_smells:
            label = f"{smell.display_name}"
            self._smell_display_to_id[label] = smell.smell_id
            values.append(label)

        self._smell_combo["values"] = values

        if values:
            if self._current_smell_id:
                # Try to preserve selection
                found = False
                for display, sid in self._smell_display_to_id.items():
                    if sid == self._current_smell_id:
                        self._smell_combo.set(display)
                        found = True
                        break
                if not found:
                    self._smell_combo.current(0)
            else:
                self._smell_combo.current(0)
            
            self._on_smell_selected()
        else:
            self._smell_combo.set("")
            self._clear_details()
            self._disable_controls(no_smells=True)

    def _on_smell_selected(self):
        display = self._smell_combo.get()
        if not display:
            return
            
        smell_id = self._smell_display_to_id.get(display)
        if not smell_id:
            return

        self._current_smell_id = smell_id
        
        # Load details
        try:
            catalog = self.catalog_service.load()
            smell = catalog.get_smell(smell_id)
            
            self._name_var.set(smell.display_name)
            self._desc_text.delete("1.0", "end")
            self._desc_text.insert("1.0", smell.description)
            
            self._enable_controls()
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel caricamento dei dettagli dello smell: {e}")

    def _clear_details(self):
        self._name_var.set("")
        self._desc_text.delete("1.0", "end")
        self._current_smell_id = None

    def _disable_controls(self, no_smells=False):
        self._name_entry.configure(state="readonly")
        self._desc_text.configure(state="disabled")
        self._save_btn.configure(state="disabled")
        self._remove_btn.configure(state="disabled")
        if no_smells:
            self._smell_combo.configure(state="disabled")
        
    def _enable_controls(self):
        self._smell_combo.configure(state="readonly")
        self._name_entry.configure(state="readonly")
        self._desc_text.configure(state="normal")
        self._save_btn.configure(state="normal")
        self._remove_btn.configure(state="normal")

    # --- Actions ---

    def _on_add_smell(self):
        # UC03 - Add Code Smell
        AddSmellDialog(self.master, self.catalog_service, self._on_smell_added_callback)

    def _on_smell_added_callback(self, new_smell_id):
        # Callback after successful add
        self._current_smell_id = new_smell_id
        self._load_smells_into_dropdown()

    def _on_remove_smell(self):
        # UC04 - Remove Code Smell
        if not self._current_smell_id:
            return

        display_name = self._name_var.get()
        confirm = messagebox.askyesno(
            "Conferma Eliminazione",
            f"Sei sicuro di voler eliminare il code smell '{display_name}'?\n\n"
            "Questa operazione non può essere annullata."
        )
        
        if confirm:
            try:
                self.catalog_service.remove_smell(self._current_smell_id)
                self._current_smell_id = None
                self._load_smells_into_dropdown()
                messagebox.showinfo("Successo", "Code smell eliminato correttamente.")
            except Exception as e:
                messagebox.showerror("Errore", f"Errore durante l'eliminazione: {e}")

    def _on_save_changes(self):
        # UC05 - Modify Code Smell
        if not self._current_smell_id:
            return

        new_desc = self._desc_text.get("1.0", "end-1c") # Remove trailing newline
        if not new_desc.strip():
            messagebox.showwarning("Attenzione", "La descrizione non può essere vuota.")
            return

        # Check if description actually changed to avoid unnecessary prompts
        try:
            catalog = self.catalog_service.load()
            current_smell = catalog.get_smell(self._current_smell_id)
            old_desc = current_smell.description
        except Exception as e:
            messagebox.showerror("Errore", f"Errore nel recupero dati originali: {e}")
            return
        
        if new_desc == old_desc:
             messagebox.showinfo("Info", "Nessuna modifica rilevata alla descrizione.")
             return

        # Confirm overwriting
        confirm = messagebox.askyesno(
            "Conferma Modifica",
            "Stai per sovrascrivere la descrizione esistente.\n"
            "La vecchia descrizione andrà persa.\n\n"
            "Vuoi proseguire con le modifiche?"
        )

        if not confirm:
            # Revert to old description
            self._desc_text.delete("1.0", "end")
            self._desc_text.insert("1.0", old_desc)
            return

        try:
            self.catalog_service.update_smell_description(self._current_smell_id, new_desc)
            messagebox.showinfo("Successo", "Descrizione aggiornata correttamente.")
            # Reload to Ensure consistency? Not strictly necessary if local update matches DB, but safe.
            self._load_smells_into_dropdown() 
        except Exception as e:
            messagebox.showerror("Errore", f"Errore durante il salvataggio: {e}")

    def _on_close(self):
        # Check for unsaved changes? 
        # For simplicity and sticking to the prompt, we trust the "Save" button. 
        # If user edits description and clicks Exit without Save, changes are lost.
        # Requirements UC03 alternate scenario mentions "exit alert" 
        # "6.a2 Sistema: Informa all’utente che se esce senza salvare, il code smell aggiunto potrebbe perdersi"
        # That refers to the ADD flow.
        # For Modify (UC05), usually "Cancel" or "Exit" implicitly discards unsaved.
        
        # We can implement a simple check if text modified vs loaded, but loading again is expensive?
        # Let's keep it simple: Just close.
        self.master.destroy()


class AddSmellDialog(tk.Toplevel):
    def __init__(self, parent, catalog_service, on_success_callback):
        super().__init__(parent)
        self.catalog_service = catalog_service
        self.on_success = on_success_callback
        
        self.title("Aggiungi Code Smell")
        self.geometry("400x300")
        self.transient(parent)
        self.grab_set()
        
        self._build_dialog_ui()
        
        # Center dialog
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")

    def _build_dialog_ui(self):
        frame = ttk.Frame(self, padding="10")
        frame.pack(fill="both", expand=True)
        frame.grid_columnconfigure(1, weight=1)

        ttk.Label(frame, text="Nome Code Smell:").grid(row=0, column=0, sticky="w", pady=5)
        self.name_entry = ttk.Entry(frame)
        self.name_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self.name_entry.focus_set()

        ttk.Label(frame, text="Descrizione:").grid(row=1, column=0, sticky="nw", pady=5)
        self.desc_text = ScrolledText(frame, height=5, width=30)
        self.desc_text.grid(row=1, column=1, sticky="nsew", pady=5)
        frame.grid_rowconfigure(1, weight=1)

        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=10, sticky="e")

        ttk.Button(btn_frame, text="Salva", command=self._on_save).pack(side="left", padx=5)
        ttk.Button(btn_frame, text="Annulla", command=self._on_cancel).pack(side="left", padx=5)

    def _on_save(self):
        name = self.name_entry.get().strip()
        desc = self.desc_text.get("1.0", "end-1c").strip()
        
        if not name:
            messagebox.showwarning("Dati mancanti", "Inserisci il nome del code smell.")
            self.name_entry.focus_set()
            return
            
        if not desc:
            messagebox.showwarning("Dati mancanti", "Inserisci una descrizione.")
            self.desc_text.focus_set()
            return

        try:
            # Check existence happens inside catalog_service.add_smell which raises CatalogValidationError
            new_id = self.catalog_service.add_smell(name, desc)
            messagebox.showinfo("Successo", "Code smell aggiunto con successo.")
            if self.on_success:
                self.on_success(new_id)
            self.destroy()
        except CatalogValidationError as e:
            messagebox.showerror("Errore Validazione", str(e))
        except Exception as e:
            messagebox.showerror("Errore", f"Impossibile salvare: {e}")

    def _on_cancel(self):
        name = self.name_entry.get().strip()
        desc = self.desc_text.get("1.0", "end-1c").strip()
        
        if name or desc:
            confirm = messagebox.askyesno(
                "Conferma Uscita", 
                "Se esci senza salvare, i dati inseriti andranno persi.\nVuoi uscire?"
            )
            if not confirm:
                return
        self.destroy()


def main():
    root = tk.Tk()
    # Apply a theme if possible to look "modern" like ttk usually does on Windows
    # style = ttk.Style()
    # style.theme_use('vista') # or 'winnative', 'clam'
    
    app = ManageCodeSmellsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
