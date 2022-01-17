
Dropzone.autoDiscover = false;

$(function() {
  var myDropzone = new Dropzone(".dropzone");
  myDropzone.on("queuecomplete", function(file) {
    // Called when all files in the queue finish uploading.
    if (myDropzone.element.id == 'songdrop') {
        window.location = "/step2";

    } else if(myDropzone.element.id == 'lrcdrop') {
        window.location = "/config";
    }
  });


})