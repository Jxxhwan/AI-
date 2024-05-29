// function previewImage(input) {
//   if (input.files && input.files[0]) {
//     let input_image_box = document.getElementById("preview");
//     let upload_btn = document.getElementById("upload_btn");
//     input_image_box.style.display = "block";
//     upload_btn.style.display = "block";

//     var reader = new FileReader();
//     reader.onload = function (e) {
//       input_image_box.setAttribute("src", e.target.result);
//     };
//     reader.readAsDataURL(input.files[0]);
//   }
// }

function previewImage(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    reader.onload = function (e) {
      var preview = document.getElementById("preview");
      preview.src = e.target.result;
      preview.style.display = "block";
    };
    reader.readAsDataURL(input.files[0]);
  }
}
