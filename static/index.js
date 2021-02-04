// window.onload = function(){
//   var elements = docoument.getElementByClassName('file-upload-image')
//   if (window.getComputedStyle(elements).display === "none") {
//     // Do something..
//   }
// };

function handleClick(cb) {
    if(cb.checked == true){
        $('.image_block').hide();
        $('.video_block').show();
    }
    else{
        $('.image_block').show();
        $('.video_block').hide();
    }
    cb.form.submit();
}

function readURL(input) {
    console.log('start')
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
        if ($('.file-upload-input').prop('type') == 'text'){
            $('.file-upload-input').atr('type', 'file')
        }

        $('.image-upload-wrap').hide();

        $('.file-upload-image').attr('src', e.target.result);
        $('.file-upload-content').show();

        $('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);
      
  } else {
    removeUpload();
  }
}

function removeUpload() {
  $('.file-upload-input').replaceWith($('.file-upload-input').clone());
  $('.file-upload-content').hide();
  $('.image-upload-wrap').show();
}


$('.image-upload-wrap').bind('dragover', function () {
		$('.image-upload-wrap').addClass('image-dropping');
	});
	$('.image-upload-wrap').bind('dragleave', function () {
		$('.image-upload-wrap').removeClass('image-dropping');
});



function readURL_vid(input) {
    console.log('start')
  if (input.files && input.files[0]) {

    var reader = new FileReader();

    reader.onload = function(e) {
        if ($('.file-upload-input').prop('type') == 'text'){
            $('.file-upload-input').atr('type', 'file')
        }
            
        $('.image-upload-wrap').hide();

        $('.file-upload-image > source').attr('src', e.target.result);
        $('.file-upload-content').show();

        $('.image-title').html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);
      
  } else {
    removeUpload();
  }
}


function loading(){
    $('.loading-page').show();
}