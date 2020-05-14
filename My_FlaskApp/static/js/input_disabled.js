$(document).ready(function () {
$( "tr" ).each(function( id ) {
    var index = id+1;
    var $inputs = $('#'+'input'+index.toString());
        $('#'+index.toString()).change(function(){
           $inputs.prop('disabled', $('#'+index.toString()).val() != 'Other');
        });
});
});