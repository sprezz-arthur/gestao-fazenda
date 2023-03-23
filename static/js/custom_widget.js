(function ($) {
    var ImageMapWidget = {
        init: function (id, options) {
            var img = $('#' + id);
            var map = img.parent().find('map');
            var points = options.value.split(',');
            var areas = [];

            $.each(points, function (index, value) {
                var coords = value.split('-');
                areas.push({
                    shape: 'circle',
                    coords: coords,
                    href: '#'
                });
            });

            img.maphilight({
                fillColor: '008800',
                fillOpacity: 0.3,
                strokeColor: 'ff0000',
                strokeOpacity: 0.5,
                strokeWidth: 2,
                alwaysOn: true,
                areas: areas
            });

            img.click(function (event) {
                var offset = img.offset();
                var x = event.pageX - offset.left;
                var y = event.pageY - offset.top;
                var coords = x + '-' + y;
                var area = {
                    shape: 'circle',
                    coords: coords,
                    href: '#'
                };
                areas.push(area);
                map.append('<area shape="' + area.shape + '" coords="' + area.coords + '" href="' + area.href + '">');
                img.maphilight({ areas: areas });
                $('#' + id + '_value').val(areas.map(function (area) { return area.coords; }).join(','));
            });
        }
    };

    window.ImageMapWidget = ImageMapWidget;
})(django.jQuery);