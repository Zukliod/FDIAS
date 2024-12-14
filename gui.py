import customtkinter as ctx
import os
from PIL import Image, ImageTk

class FDIAS(ctx.CTk):
    def __init__(self):
        super().__init__()     
        ctx.set_appearance_mode('dark')
        self.title("FDIAS - Face Detection & Intrusion Alert System")

        self.image_frame = ctx.CTkFrame(self)
        self.image_frame.pack(side="top", pady=10, padx=10, fill="both", expand=True)

        self.button = ctx.CTkButton(self, text="Add Member", command=self.upload_image)
        self.button.pack(side="bottom", pady=10, padx=10, fill="x")

        self.image_vars = [] 
        self.update_image_grid()
        # self.geometry(f'{width}x{height}')
        self.mainloop()

    def delete_image(self, image_path):
        os.remove(image_path)
        directory = os.path.dirname(image_path)
        if not os.listdir(directory): os.rmdir(directory)

        self.update_image_grid()

    def update_image_grid(self):
        image_dir = "./storage/faces/"
        row, col = 0, 0

        for widget in self.image_frame.winfo_children():
            widget.destroy()

        for directory in os.listdir(image_dir):
            if("." in directory): continue
            
            for image in os.listdir(os.path.join(image_dir, directory)):
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(image_dir, directory, image)
                    img = Image.open(image_path)

                    width, height = img.size
                    target_size = 100

                    if width > height:
                        new_width = width * (target_size / height)
                        new_height = target_size
                    else:
                        new_width = target_size
                        new_height = height * (target_size / width)

                    img = img.resize((int(new_width), int(new_height)))
                    left = (new_width - target_size) / 2
                    top = (new_height - target_size) / 2
                    img = img.crop((left, top, left + target_size, top + target_size))
                    photo = ImageTk.PhotoImage(img)
                    self.image_vars.append(photo)  # Store reference to avoid garbage collection

                    frame = ctx.CTkFrame(self.image_frame, corner_radius=0)
                    frame.grid(row=row, column=col, padx=5, pady=5)

                    name_label = ctx.CTkLabel(frame, text=directory.upper())
                    name_label.pack(anchor="w", padx=5)

                    image_label = ctx.CTkLabel(frame, image=photo, text="")
                    image_label.pack()

                    delete_button = ctx.CTkButton(frame, corner_radius=0, text="Delete", command=lambda p=image_path: self.delete_image(p), width=100)
                    delete_button.pack(padx=5, pady=5)

                    col += 1
                    if col >= 4:
                        col = 0
                        row += 1

                    break

    def upload_image(self):
        file_path = ctx.filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])

        if file_path:
            name = ctx.CTkInputDialog(title="Enter name", text="Please enter the name of the person:")
            person_name = name.get_input().lower()
            person_dir = os.path.join("./faces", person_name)
            os.makedirs(person_dir, exist_ok=True)
            image = Image.open(file_path)
            image.save(os.path.join(person_dir, os.path.basename(file_path)))

            self.update_image_grid()

if __name__ == "__main__":
    app = FDIAS()