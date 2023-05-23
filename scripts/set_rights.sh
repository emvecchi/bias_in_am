DATA=/mount/arbeitsdaten14/projekte/sfb-732/d8/falensaa/BiasInArguments/

chgrp -R imsstaff $DATA
chmod -R g+w $DATA
chmod -R g+r $DATA
chmod -R o+r $DATA

for directory in $(find $DATA -type d); do
    chmod 775 $directory
done

for filename in $(find $DATA/scripts -type f -name "*"); do
    chmod 775 $filename
done


