function ansave(fig, name)
    if nargin == 0; fig = gcf; end
    if nargin <= 1; name = fig.FileName; name = name(1:end-4); end

    change_to_dark = strcmp(fig.Theme.BaseColorStyle, "dark");
    if change_to_dark
        theme(fig,"light");
    end
    saveas(fig, append(name, '.png'));
    saveas(fig, append(name, '.fig'));
    if change_to_dark
        theme(fig,"dark");
    end
end